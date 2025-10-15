"""
Local embedding generation using Qwen3-Embedding-0.6B with CUDA acceleration.
Supports both local transformers-based inference and vLLM server-based inference.
"""
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
from openai import OpenAI


class LocalEmbeddings:
    """Wrapper for Qwen3-Embedding-0.6B model to generate embeddings locally using CUDA."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cuda", use_multi_gpu: bool = True):
        """
        Initialize the local embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cuda' or 'cpu')
            use_multi_gpu: If True and multiple GPUs available, use specific GPU to avoid conflicts
        """
        self.device = device
        self.max_length = 8192

        print(f"Loading embedding model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        # For small embedding model (~2GB), use single GPU
        # Fits easily on GPU 0 alongside vLLM instance (~24GB used, 80GB available)
        if device == "cuda":
            embed_device = "cuda:0"  # Share GPU 0 with one vLLM instance
            print(f"   Using GPU 0 for embeddings (~2GB, shares with vLLM instance)")
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.float16
            ).to(embed_device)
            self.device = embed_device
            self.multi_gpu = False
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            self.multi_gpu = False

        self.model.eval()
        print(f"Embedding model loaded successfully on {self.device}")

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Pool the last token from the hidden states.

        Args:
            last_hidden_states: Hidden states from the model
            attention_mask: Attention mask for the input

        Returns:
            Pooled embeddings
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2 normalize the embeddings

        Returns:
            Embeddings as list(s) of floats
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        # Tokenize
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device (for multi-GPU with device_map, this is handled automatically)
        if not self.multi_gpu:
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        else:
            # For device_map="auto", move to the first device (model handles distribution)
            batch_dict = {k: v.to("cuda:0") for k, v in batch_dict.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )

            # Normalize if requested
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert to list format
        embeddings_list = embeddings.cpu().float().numpy().tolist()

        if return_single:
            return embeddings_list[0]
        return embeddings_list

    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        # For Qwen3-Embedding-0.6B, the embedding dimension is 1024
        return self.model.config.hidden_size


class VLLMEmbeddings:
    """Wrapper for vLLM embedding server using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    ):
        """
        Initialize the vLLM embedding client.

        Args:
            base_url: Base URL for the vLLM embedding server (e.g., http://localhost:8001/v1)
            api_key: API key for authentication
            model_name: Name of the embedding model being served
        """
        self.base_url = base_url or os.getenv("VLLM_EMBEDDING_BASE_URL", "http://localhost:8001/v1")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "token-abc123")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

        print(f"Initializing vLLM embedding client for {self.model_name}...")
        print(f"   Connecting to: {self.base_url}")

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Cache the embedding dimension (1024 for Qwen3-Embedding-0.6B)
        self._embedding_dim = 1024 if "0.6B" in self.model_name else None

        print(f"vLLM embedding client initialized successfully")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s) into embeddings using the vLLM server.

        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2 normalize the embeddings (handled by vLLM server)

        Returns:
            Embeddings as list(s) of floats
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        # Call vLLM embedding API
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]

        if return_single:
            return embeddings[0]
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        if self._embedding_dim is not None:
            return self._embedding_dim

        # If not cached, get it from a test embedding
        test_embedding = self.encode("test", normalize=True)
        self._embedding_dim = len(test_embedding)
        return self._embedding_dim


# Singleton instance for reuse
_embedding_model = None


def get_embedding_model(
    device: str = "cuda",
    use_multi_gpu: bool = True,
    use_vllm: bool = None,
) -> Union[LocalEmbeddings, VLLMEmbeddings]:
    """
    Get or create a singleton instance of the embedding model.

    Args:
        device: Device to run the model on (only used for LocalEmbeddings)
        use_multi_gpu: If True, use all available GPUs (only used for LocalEmbeddings)
        use_vllm: If True, use vLLM server; if False, use local transformers.
                  If None, read from USE_VLLM_EMBEDDINGS env variable (default: true)

    Returns:
        LocalEmbeddings or VLLMEmbeddings instance
    """
    global _embedding_model
    if _embedding_model is None:
        # Determine whether to use vLLM or local transformers
        if use_vllm is None:
            use_vllm = os.getenv("USE_VLLM_EMBEDDINGS", "true").lower() == "true"

        if use_vllm:
            print("Using vLLM embedding server (DP=8 across GPUs)")
            _embedding_model = VLLMEmbeddings()
        else:
            print("Using local transformers embedding model")
            _embedding_model = LocalEmbeddings(device=device, use_multi_gpu=use_multi_gpu)

    return _embedding_model
