import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from torch.utils.data import DataLoader, Dataset

MODEL_NAME = os.environ.get("MODEL_NAME", "VLLM_MODEL_NAME")

VLLM_HOST = os.environ.get("VLLM_HOST", "http://VLLM_HOST:VLLM_PORT")
API_ENDPOINT = f"{VLLM_HOST}/v1/chat/completions"
API_KEY = os.environ.get("API_KEY", "token-abc123")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 4096))

df_questions = pd.read_json("questions.jsonl", orient='records', encoding="utf-8-sig", lines=True)


class QuestionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


def collate_fn(batch):
    return pd.DataFrame(batch)


def request_with_messages(messages, max_retries=20):
    payload = {
        "messages": messages,
        "model": MODEL_NAME,
        "frequency_penalty": 0,
        "max_tokens": MAX_TOKENS,
        "presence_penalty": 0,
        "stream": False,
        "temperature": 0,
        "top_p": 1,
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(
                API_ENDPOINT, json=payload, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError) as e:
            print(f"Error: {e}")
            retries += 1
            time.sleep(2)  # wait for 2 seconds before retrying
    return "Failed to generate response after several attempts."


def process_batch(batch):
    single_turn_outputs = []

    for question in batch["questions"]:
        messages = [
            {"role": "user", "content": question[0]},
        ]
        output = request_with_messages(messages)
        single_turn_outputs.append(output)

    multi_turn_questions = []
    for idx, row in batch.iterrows():
        multi_turn_prompt = [
            {"role": "user", "content": row["questions"][0]},
            {"role": "assistant", "content": single_turn_outputs[0]},
            {"role": "user", "content": row["questions"][1]},
        ]
        multi_turn_questions.append(multi_turn_prompt)

    multi_turn_outputs = []
    for prompt in multi_turn_questions:
        multi_turn_outputs.append(prompt)

    return pd.DataFrame(
        {
            "id": batch["id"],
            "category": batch["category"],
            "questions": batch["questions"],
            "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
            "references": batch["references"],
        }
    )


def process_data(df_questions, batch_size=1, num_workers=42):
    dataset = QuestionDataset(df_questions)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, dataloader))

    df_output = pd.concat(results, ignore_index=True)
    output_json = f"{MODEL_NAME}.jsonl"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    df_output.to_json(output_json, orient="records", lines=True, force_ascii=False)


# Call the process_data function with appropriate parameters
process_data(df_questions, batch_size=1, num_workers=42)
