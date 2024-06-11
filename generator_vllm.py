import os
import pandas as pd
import requests
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
import time

API_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "MODEL_NAME_HERE"
API_KEY = 'YOUR_API_KEY'

df_questions = pd.read_json('questions.jsonl', lines=True)

class QuestionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

def collate_fn(batch):
    return pd.DataFrame(batch)

def generate(prompt, max_retries=20):
    payload = {
        "messages": [
            {"content": "You are a helpful assistant", "role": "system"},
            {"content": prompt, "role": "user"}
        ],
        "model": MODEL_NAME,
        "frequency_penalty": 0,
        "max_tokens": 8192,
        "presence_penalty": 0,
        "stream": False,
        "temperature": 0,
        "top_p": 1,
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(API_ENDPOINT, json=payload, headers={'Authorization': f'Bearer {API_KEY}'}, timeout=120)
            response.raise_for_status()
            result = response.json()
            print(result)
            return result['choices'][0]['message']["content"].strip()
        except (requests.RequestException, KeyError) as e:
            print(f"Error: {e}")
            retries += 1
            time.sleep(10)  # wait for 2 seconds before retrying
    return "Failed to generate response after several attempts."

def process_batch(batch):
    print(batch)
    single_turn_outputs = []

    for question in batch['questions']:
        output = generate(question[0])
        single_turn_outputs.append(output)

    multi_turn_questions = []
    for idx, row in batch.iterrows():
        multi_turn_prompt = [
            {"role": "user", "content": row['questions'][0]},
            {"role": "assistant", "content": single_turn_outputs[0]},
            {"role": "user", "content": row['questions'][1]}
        ]
        print("multi")
        multi_turn_questions.append(multi_turn_prompt)

    multi_turn_outputs = []
    for prompt in multi_turn_questions:
        payload = {
            "messages": prompt,
            "model": MODEL_NAME,
            "frequency_penalty": 0,
            "max_tokens": 8192,
            "presence_penalty": 0,
            "stream": False,
            "temperature": 0,
            "top_p": 1,
        }
        retries = 0
        while retries < 5:
            try:
                response = requests.post(API_ENDPOINT, json=payload, headers={'Authorization': f'Bearer {API_KEY}'}, timeout=120)
                response.raise_for_status()
                result = response.json()
                
                multi_turn_outputs.append(result['choices'][0]['message']["content"].strip())
                break
            except (requests.RequestException, KeyError) as e:
                print(f"Error: {e}")
                retries += 1
                time.sleep(2)
        else:
            multi_turn_outputs.append("Failed to generate response after several attempts.")

    return pd.DataFrame({
        'id': batch['id'],
        'category': batch['category'],
        'questions': batch['questions'],
        'outputs': list(zip(single_turn_outputs, multi_turn_outputs)),
        'references': batch['references']
    })

def process_data(df_questions, batch_size=1, num_workers=42):
    dataset = QuestionDataset(df_questions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, dataloader))

    df_output = pd.concat(results, ignore_index=True)
    df_output.to_json(
        f'{MODEL_NAME}.jsonl',
        orient='records',
        lines=True,
        force_ascii=False
    )

# Call the process_data function with appropriate parameters
process_data(df_questions, batch_size=1, num_workers=42)
