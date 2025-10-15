import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict

# Python 3.10 compatibility patch for typing.NotRequired
import typing

if not hasattr(typing, "NotRequired"):
    from typing_extensions import NotRequired

    typing.NotRequired = NotRequired

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool, create_search_memory_tool
from openai import OpenAI
from prompts import ANSWER_PROMPT
from tqdm import tqdm
from .local_embeddings import get_embedding_model

load_dotenv()

# Configure OpenAI client to use local vLLM server
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# Initialize local embedding model (singleton) with all GPUs
embedding_model = get_embedding_model(device="cuda", use_multi_gpu=True)

ANSWER_PROMPT_TEMPLATE = Template(ANSWER_PROMPT)


def get_answer(question, speaker_1_user_id, speaker_1_memories, speaker_2_user_id, speaker_2_memories):
    prompt = ANSWER_PROMPT_TEMPLATE.render(
        question=question,
        speaker_1_user_id=speaker_1_user_id,
        speaker_1_memories=speaker_1_memories,
        speaker_2_user_id=speaker_2_user_id,
        speaker_2_memories=speaker_2_memories,
    )

    t1 = time.time()
    response = client.chat.completions.create(
        model=os.getenv("MODEL"), messages=[{"role": "system", "content": prompt}], temperature=0.0
    )
    t2 = time.time()
    return response.choices[0].message.content, t2 - t1


def prompt(state):
    """Prepare the messages for the LLM."""
    store = get_store()
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    # Use trimmed messages if available (from pre_model_hook), otherwise fall back to full messages
    messages = state.get("llm_input_messages", state["messages"])
    return [{"role": "system", "content": system_msg}, *messages]


class LangMem:
    def __init__(
        self,
    ):
        # Custom embedding function using local Qwen3-Embedding-0.6B model
        def embed_texts(texts: list[str]) -> list[list[float]]:
            """Embed texts using local Qwen3-Embedding model."""
            return embedding_model.encode(texts, normalize=True)

        # Get embedding dimensions from the model
        embedding_dims = embedding_model.get_embedding_dim()

        self.store = InMemoryStore(
            index={
                "dims": embedding_dims,  # Qwen3-Embedding-0.6B has 1024 dims
                "embed": embed_texts,
            }
        )
        self.checkpointer = MemorySaver()  # Checkpoint graph state

        # Trim messages to prevent context overflow (this is needed but can experiment with max_tokens)
        def pre_model_hook(state):
            trimmed_messages = trim_messages(
                state["messages"],
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=24576,  # 24k, leave 8k for agent
                start_on="human",
                end_on=("human", "tool"),
            )
            return {"llm_input_messages": trimmed_messages}

        self.agent = create_react_agent(
            f"openai:{os.getenv('MODEL')}",
            prompt=prompt,
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=self.store,
            checkpointer=self.checkpointer,
            pre_model_hook=pre_model_hook,
        )

    def add_memory(self, message, config):
        # prevent context window overflow (in this case, just catch the error and do nothing)
        try:
            result = self.agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
        except Exception as e:
            print(f"Error in add_memory: {e}")
            return ""
        return result

    def search_memory(self, query, config):
        try:
            t1 = time.time()
            response = self.agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
            t2 = time.time()
            return response["messages"][-1].content, t2 - t1
        except Exception as e:
            print(f"Error in search_memory: {e}")
            return "", 0


class LangMemManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        with open(self.dataset_path, "r") as f:
            self.data = json.load(f)

    def process_all_conversations(self, output_file_path):
        OUTPUT = defaultdict(list)

        # Process conversations in parallel with multiple workers
        def process_conversation(key_value_pair):
            key, value = key_value_pair
            result = defaultdict(list)

            chat_history = value["conversation"]
            questions = value["question"]

            agent1 = LangMem()
            agent2 = LangMem()
            max_iterations = 30
            recursion_limit = 2 * max_iterations + 1
            config = {"configurable": {"thread_id": f"thread-{key}"}, "recursion_limit": recursion_limit}

            speakers = set()

            # Identify speakers
            for conv in chat_history:
                speakers.add(conv["speaker"])

            if len(speakers) != 2:
                raise ValueError(f"Expected 2 speakers, got {len(speakers)}")

            speaker1 = list(speakers)[0]
            speaker2 = list(speakers)[1]

            # Add memories for each message
            for conv in tqdm(chat_history, desc=f"Processing messages in thread {key}", leave=False):
                message = f"{conv['timestamp']} | {conv['speaker']}: {conv['text']}"
                if conv["speaker"] == speaker1:
                    agent1.add_memory(message, config)
                elif conv["speaker"] == speaker2:
                    agent2.add_memory(message, config)
                else:
                    raise ValueError(f"Expected speaker1 or speaker2, got {conv['speaker']}")

            # Process questions
            for q in tqdm(questions, desc=f"Processing questions in thread {key}", leave=False):
                category = q["category"]

                if int(category) == 5:
                    continue

                answer = q["answer"]
                question = q["question"]
                response1, speaker1_memory_time = agent1.search_memory(question, config)
                response2, speaker2_memory_time = agent2.search_memory(question, config)

                generated_answer, response_time = get_answer(question, speaker1, response1, speaker2, response2)

                result[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "response1": response1,
                        "response2": response2,
                        "category": category,
                        "speaker1_memory_time": speaker1_memory_time,
                        "speaker2_memory_time": speaker2_memory_time,
                        "response_time": response_time,
                        "response": generated_answer,
                    }
                )

            return result

        # Use multiprocessing to process conversations in parallel
        # Use threads instead of processes (no pickling issues, great for I/O/API bound work)
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            for item in self.data.items():
                futures.append(ex.submit(process_conversation, item))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing conversations"):
                results.append(fut.result())

                # Combine results from all workers
                for result in results:
                    for key, items in result.items():
                        OUTPUT[key].extend(items)

                # Save final results
                with open(output_file_path, "w") as f:
                    json.dump(OUTPUT, f, indent=4)
