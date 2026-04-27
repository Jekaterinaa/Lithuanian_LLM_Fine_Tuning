from pydantic import BaseModel, Field
from typing import List, Literal
import random
import json
from multiprocessing import Pool
from ollama import chat
from prompts.system_prompts import system_prompt_grammar, system_prompt_qa, system_prompt_maironis
from prompts.user_prompts import user_prompt_grammar, user_prompt_qa, user_prompt_maironis


MODEL = "qwen2.5:32b"
BATCH_SIZE = 5
QA_TOTAL = 5000
GRAMMAR_TOTAL = 2500
MAIRONIS_TOTAL = 2500
NUM_WORKERS = 4

ERROR_TYPES = ["case", "verb", "agreement", "word_form", "preposition"]

TOPICS = [
    "Artificial Intelligence", "Lithuanian History", "Space Exploration",
    "Health and Medicine", "Climate Change", "Education Systems",
    "Sports and Athletics", "Music and Art", "Food and Cuisine",
    "Travel and Tourism", "Economics and Finance", "Philosophy",
    "Mathematics", "Geography", "Psychology",
    "Biology and Nature", "Physics", "Literature",
    "Technology and Innovation", "Politics and Society"
]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GrammarItem(BaseModel):
    incorrect: str
    correct: str


class GrammarResponse(BaseModel):
    examples: List[GrammarItem]


class QAItem(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    examples: List[QAItem]


class MaironisItem(BaseModel):
    input: str
    output: str


class MaironisResponse(BaseModel):
    examples: List[MaironisItem]


class FineTuningExample(BaseModel):
    messages: List[Message]


def to_fine_tuning(query: str, response: str) -> FineTuningExample:
    return FineTuningExample(messages=[
        Message(role="user", content=query),
        Message(role="assistant", content=response),
    ])


def generate_grammar_batch(error_type: str) -> List[dict]:
    """Generate a single batch of grammar examples for a given error type."""
    system_msg = {"role": "system", "content": system_prompt_grammar}
    user_msg = {"role": "user", "content": user_prompt_grammar.format(n=BATCH_SIZE, error_type=error_type)}

    try:
        response = chat(
            model=MODEL,
            messages=[system_msg, user_msg],
            format=GrammarResponse.model_json_schema()
        )
        grammar_data = GrammarResponse.model_validate_json(response.message.content)
        results = [
            to_fine_tuning(query=item.incorrect, response=item.correct).model_dump()
            for item in grammar_data.examples
        ]
        print(f"[Grammar/{error_type}] Generated {len(results)} examples")
        return results
    except Exception as e:
        print(f"[Grammar/{error_type}] Error: {e}")
        return []


def generate_qa_batch(topic: str) -> List[dict]:
    """Generate a single batch of QA examples for a given topic."""
    system_msg = {"role": "system", "content": system_prompt_qa}
    user_msg = {"role": "user", "content": user_prompt_qa.format(n=BATCH_SIZE, topic=topic)}

    try:
        response = chat(
            model=MODEL,
            messages=[system_msg, user_msg],
            format=QAResponse.model_json_schema()
        )
        qa_data = QAResponse.model_validate_json(response.message.content)
        results = [
            to_fine_tuning(query=item.question, response=item.answer).model_dump()
            for item in qa_data.examples
        ]
        print(f"[QA/{topic}] Generated {len(results)} examples")
        return results
    except Exception as e:
        print(f"[QA/{topic}] Error: {e}")
        return []


def generate_maironis_batch(_: int) -> List[dict]:
    """Generate a single batch of Maironis examples."""
    system_msg = {"role": "system", "content": system_prompt_maironis}
    user_msg = {"role": "user", "content": user_prompt_maironis.format(n=BATCH_SIZE)}

    try:
        response = chat(
            model=MODEL,
            messages=[system_msg, user_msg],
            format=MaironisResponse.model_json_schema()
        )
        maironis_data = MaironisResponse.model_validate_json(response.message.content)
        results = [
            to_fine_tuning(query=item.input, response=item.output).model_dump()
            for item in maironis_data.examples
        ]
        print(f"[Maironis] Generated {len(results)} examples")
        return results
    except Exception as e:
        print(f"[Maironis] Error: {e}")
        return []


def build_tasks():
    """Build all task lists for parallel execution."""
    tasks = []

    # Grammar: 2500 total / 5 error types = 500 per type / 5 batch = 100 batches per type
    batches_per_error_type = GRAMMAR_TOTAL // (len(ERROR_TYPES) * BATCH_SIZE)
    for error_type in ERROR_TYPES:
        for _ in range(batches_per_error_type):
            tasks.append(("grammar", error_type))

    # QA: 5000 total / 20 topics = 250 per topic / 5 batch = 50 batches per topic
    batches_per_topic = QA_TOTAL // (len(TOPICS) * BATCH_SIZE)
    for topic in TOPICS:
        for _ in range(batches_per_topic):
            tasks.append(("qa", topic))

    # Maironis: 2500 total / 5 batch = 500 batches
    maironis_batches = MAIRONIS_TOTAL // BATCH_SIZE
    for i in range(maironis_batches):
        tasks.append(("maironis", i))

    random.shuffle(tasks)
    return tasks


def process_task(task: tuple) -> List[dict]:
    """Dispatch a single task to the appropriate generator."""
    task_type, param = task
    if task_type == "grammar":
        return generate_grammar_batch(param)
    elif task_type == "qa":
        return generate_qa_batch(param)
    elif task_type == "maironis":
        return generate_maironis_batch(param)
    return []


if __name__ == "__main__":
    tasks = build_tasks()
    print(f"Total tasks to process: {len(tasks)}")
    print(f"Expected examples: ~{len(tasks) * BATCH_SIZE}")

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_task, tasks)

    # Flatten results
    examples = [ex for batch in results for ex in batch]
    print(f"\nTotal examples generated: {len(examples)}")

    random.shuffle(examples)

    with open("dataset/fine_tuning_dataset.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(examples)} examples to dataset/fine_tuning_dataset.json")
