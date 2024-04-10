import os
import argparse
import pandas as pd
import requests
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--template', help=' : Template File Location', default='./templates/template-EEVE.json')
parser.add_argument('--batch_size', help=' : Batch Size', default=2, type=int)
parser.add_argument('--num_workers', help=' : Number of DataLoader Workers', default=2, type=int)
args = parser.parse_args()

df_config = pd.read_json(args.template, typ='series')
SINGLE_TURN_TEMPLATE = df_config.iloc[0]  
MULTI_TURN_TEMPLATE = df_config.iloc[1]

API_ENDPOINT = "{YOUR_VLLM_ENDPOINT}/v1/completions"

df_questions = pd.read_json('questions.jsonl', lines=True)

def format_single_turn_question(question):
    return SINGLE_TURN_TEMPLATE.format(question[0])

class QuestionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  
        return self.df.iloc[idx]

def collate_fn(batch):
    return pd.DataFrame(batch)

def process_batch(batch):
    single_turn_questions = batch['questions'].apply(lambda x: format_single_turn_question(x))

    def generate(prompt):
        payload = {
            "model": "{YOUR_MODEL}",
            "max_tokens": 4096,
            "temperature": 0,
            "top_p" : 1,
            "top_k" : -1,
            "early_stopping" : True,
            "best_of" : 4,
            "use_beam_search" : True,
            "skip_special_tokens" : False,
            "prompt" : prompt
        }

        response = requests.post(API_ENDPOINT, json=payload)
        result = response.json()
        print(prompt)
        print(result)
        return result['choices'][0]['text'].strip().replace("<|im_end|>","")

    single_turn_outputs = []
    s = 0

    for prompt in single_turn_questions.tolist():
        output = generate(prompt)
        single_turn_outputs.append(output)
        s = s + 1
        print("s " + str(s))

    def format_multi_turn_question(row):  
        return MULTI_TURN_TEMPLATE.format(
            row['questions'][0], single_turn_outputs[row.name], row['questions'][1]  
        )

    multi_turn_questions = batch.apply(format_multi_turn_question, axis=1)

    multi_turn_outputs = []
    i = 0

    for prompt in multi_turn_questions.tolist():
        output = generate(prompt)
        multi_turn_outputs.append(output)
        i = i + 1
        print("m " + str(i))

    return pd.DataFrame({
        'id': batch['id'],
        'category': batch['category'], 
        'questions': batch['questions'],
        'outputs': list(zip(single_turn_outputs, multi_turn_outputs)),
        'references': batch['references']
    })

def process_data(df_questions, batch_size, num_workers):
    dataset = QuestionDataset(df_questions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=None,
        pin_memory=True
    )

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, dataloader))

    df_output = pd.concat(results, ignore_index=True)
    df_output.to_json(
        'qwen1.5-32B-Chat.jsonl',
        orient='records', 
        lines=True,
        force_ascii=False
    )

process_data(df_questions, args.batch_size, args.num_workers)
