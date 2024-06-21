import argparse
import pandas as pd
from anthropic import Anthropic
import time
import os
from tqdm import tqdm

MAX_MODEL_LEN = 4096
MODEL = "claude-3-5-sonnet-20240620"  # Update this to the appropriate Anthropic model

client = Anthropic(api_key="...")  # Replace with your Anthropic API key

df_questions = pd.read_json('questions.jsonl', lines=True)

def format_single_turn_question(question):
    return question[0]

single_turn_questions = df_questions['questions'].map(format_single_turn_question)
single_turn_outputs = []

for question in tqdm(single_turn_questions, desc="Processing single-turn questions"):
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_MODEL_LEN,
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    single_turn_outputs.append(response.content[0].text)

def format_double_turn_question(question, single_turn_output):
    return [question[0], single_turn_output, question[1]]

multi_turn_questions = df_questions[['questions', 'id']].apply(lambda x: format_double_turn_question(x['questions'], single_turn_outputs[x['id']-1]), axis=1)
multi_turn_outputs = []

for question in tqdm(multi_turn_questions, desc="Processing multi-turn questions"):
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_MODEL_LEN,
        messages=[
            {"role": "user", "content": question[0]},
            {"role": "assistant", "content": question[1]},
            {"role": "user", "content": question[2]},
        ],
        temperature=0,
    )
    multi_turn_outputs.append(response.content[0].text)

df_output = pd.DataFrame({'id': df_questions['id'], 'category': df_questions['category'], 'questions': df_questions['questions'], 'outputs': list(zip(single_turn_outputs, multi_turn_outputs)), "references": df_questions['references']})
df_output.to_json(f'{str(MODEL).replace("/", "_")}.jsonl', orient='records', lines=True, force_ascii=False)