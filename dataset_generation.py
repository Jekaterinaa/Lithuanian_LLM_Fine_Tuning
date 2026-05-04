from pydantic import BaseModel, Field
from typing import List, Literal
import random
import json
from ollama import chat
from prompts.system_prompts import system_prompt_grammar, system_prompt_qa, system_prompt_maironis
from prompts.user_prompts import user_prompt_grammar, user_prompt_qa, user_prompt_maironis

# Larger model
MODEL = "qwen2.5:32b"
BATCH_SIZE = 5

# examples to generate
GRAMMAR_TOTAL = 250
QA_TOTAL = 500
MAIRONIS_TOTAL = 250


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


def generate_example(messages: List[Message], model: str, response_format: BaseModel):
    return chat(
        model=model,
        messages=messages,
        format=response_format.model_json_schema()
    )


def to_fine_tuning(query: str, response: str) -> FineTuningExample:
    return FineTuningExample(messages=[
        Message(role="user", content=query),
        Message(role="assistant", content=response),
    ])


if __name__ == "__main__":

    examples = []


    # Grammar
    batches_per_error_type = GRAMMAR_TOTAL // (len(ERROR_TYPES) * BATCH_SIZE)

    for error_type in ERROR_TYPES:
        print(f"\n[Grammar] Generating {batches_per_error_type} batches for error type: {error_type}")
        for i in range(batches_per_error_type):
            try:
                system_msg = {"role": "system", "content": system_prompt_grammar}
                user_msg = {"role": "user", "content": user_prompt_grammar.format(n=BATCH_SIZE, error_type=error_type)}

                response = generate_example([system_msg, user_msg], MODEL, GrammarResponse)
                grammar_data = GrammarResponse.model_validate_json(response.message.content)

                batch = [
                    to_fine_tuning(query=item.incorrect, response=item.correct)
                    for item in grammar_data.examples
                ]
                examples.extend(batch)
                print(f"  Batch {i + 1}/{batches_per_error_type} — {len(batch)} examples (total: {len(examples)})")
            except Exception as e:
                print(f"  Batch {i + 1}/{batches_per_error_type} — ERROR: {e}")


    # QA
    batches_per_topic = QA_TOTAL // (len(TOPICS) * BATCH_SIZE)  # 50

    for topic in TOPICS:
        print(f"\n[QA] Generating {batches_per_topic} batches for topic: {topic}")
        for i in range(batches_per_topic):
            try:
                system_msg = {"role": "system", "content": system_prompt_qa}
                user_msg = {"role": "user", "content": user_prompt_qa.format(n=BATCH_SIZE, topic=topic)}

                response = generate_example([system_msg, user_msg], MODEL, QAResponse)
                qa_data = QAResponse.model_validate_json(response.message.content)

                batch = [
                    to_fine_tuning(query=item.question, response=item.answer)
                    for item in qa_data.examples
                ]
                examples.extend(batch)
                print(f"  Batch {i + 1}/{batches_per_topic} — {len(batch)} examples (total: {len(examples)})")
            except Exception as e:
                print(f"  Batch {i + 1}/{batches_per_topic} — ERROR: {e}")


    # Poetic style
    maironis_batches = MAIRONIS_TOTAL // BATCH_SIZE  # 500

    print(f"\n[Maironis] Generating {maironis_batches} batches")
    for i in range(maironis_batches):
        try:
            system_msg = {"role": "system", "content": system_prompt_maironis}
            user_msg = {"role": "user", "content": user_prompt_maironis.format(n=BATCH_SIZE)}

            response = generate_example([system_msg, user_msg], MODEL, MaironisResponse)
            maironis_data = MaironisResponse.model_validate_json(response.message.content)

            batch = [
                to_fine_tuning(query=item.input, response=item.output)
                for item in maironis_data.examples
            ]
            examples.extend(batch)
            print(f"  Batch {i + 1}/{maironis_batches} — {len(batch)} examples (total: {len(examples)})")
        except Exception as e:
            print(f"  Batch {i + 1}/{maironis_batches} — ERROR: {e}")


    # --- Shuffle and save ---
    print(f"\nGeneration complete. Total examples: {len(examples)}")
    random.shuffle(examples)

    with open("dataset/fine_tuning_dataset_with_32b.json", "w", encoding="utf-8") as f:
        json.dump(
            [ex.model_dump() for ex in examples],
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved {len(examples)} examples to dataset/fine_tuning_dataset_with_32b.json")
