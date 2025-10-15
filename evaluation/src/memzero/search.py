import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

from mem0 import MemoryClient

load_dotenv()

# Configure OpenAI client to use local vLLM server
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")


class MemorySearch:
    def __init__(self, output_path="mem0_results.json", top_k=10, filter_memories=False, is_graph=False):
        self.mem0_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        self.top_k = top_k
        self.openai_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self._save_lock = threading.Lock()  # Lock for thread-safe file writes

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    print("Searching with graph")
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        top_k=self.top_k,
                        filter_memories=self.filter_memories,
                        enable_graph=True,
                        output_format="v1.1",
                    )
                else:
                    memories = self.mem0_client.search(
                        query, user_id=user_id, top_k=self.top_k, filter_memories=self.filter_memories
                    )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, save_after=True):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        # Save results after each question is processed (only if save_after is True)
        if save_after:
            with self._save_lock:
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path, max_workers=1):
        """Process all conversations from a data file.

        Args:
            file_path: Path to the JSON data file
            max_workers: Number of parallel workers for question processing.
                        If 1, processes sequentially. If >1, uses parallel processing.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            if max_workers > 1:
                # Use parallel processing
                self.process_questions_parallel(idx, qa, speaker_a_user_id, speaker_b_user_id, max_workers)
            else:
                # Sequential processing (original behavior)
                for question_item in tqdm(
                    qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
                ):
                    result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                    self.results[idx].append(result)

                    # Save results after each question is processed
                    with open(self.output_path, "w") as f:
                        json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, idx, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=10):
        """Process questions in parallel for a single conversation."""

        def process_single_question(val):
            # Don't save after each question in parallel mode to avoid file write contention
            return self.process_question(val, speaker_a_user_id, speaker_b_user_id, save_after=False)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_question, qa_list),
                    total=len(qa_list),
                    desc=f"Processing questions for conversation {idx}",
                    leave=False,
                )
            )

        # Add all results to self.results[idx]
        self.results[idx].extend(results)

        # Save after processing all questions for this conversation
        with self._save_lock:
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        return results
