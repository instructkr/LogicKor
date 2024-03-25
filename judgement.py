import argparse
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
import time
from datetime import datetime
import re

time_start = datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--model-output', help=' : Model Output File Location', default=None)
parser.add_argument('--openai-api-key', help=' : Model', default=None)
parser.add_argument('--judge-model', help=' : Judge Model', default='gpt-4-1106-preview')
parser.add_argument('--threads', help=' : Thread count', default=10, type=int)
args = parser.parse_args()

if args.model_output is None:
    raise ValueError('Model Output File Location is required')
if args.openai_api_key is None:
    raise ValueError('OpenAI API Key is required')

client = OpenAI(
  api_key=args.openai_api_key
)

df_model_outputs = pd.read_json(args.model_output, lines=True)
df_judge_template = pd.read_json('judge_template.jsonl', lines=True)

lock = Lock()
def create_answers(model_output, is_multi_turn=False):
    # Construct prompt from model output
    prompt = f"**질문**\n{model_output['questions'][0]}\n\n**모델 답변**\n{model_output['outputs'][0]}"
    if model_output['references'] and model_output['references'][0]:
        prompt += f"\n\n**Ground Truth**\n{model_output['references'][0]}"

    if is_multi_turn:
        prompt += f"\n\n**이어지는 질문**\n{model_output['questions'][1]}\n\n**모델 답변**\n{model_output['outputs'][1]}"
        if model_output['references'] and model_output['references'][1]:
            prompt += f"\n\n**Ground Truth**\n{model_output['references'][1]}"
    prompt += "\n\n[[대화 종료. 평가 시작.]]"

    try:
        response = client.chat.completions.create(
            model=args.judge_model,
            temperature=0,
            n=1,
            messages=[
                {"role": "system", "content": df_judge_template.iloc[1 if is_multi_turn else 0]['system_prompt']},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract judge message and score using regular expressions
        content = response.choices[0].message.content
        judge_message_match = re.search(r"평가:(.*?)점수:", content, re.DOTALL)
        judge_message = judge_message_match.group(1).strip() if judge_message_match else "No judge message found"
        
        judge_score_match = re.search(r"점수:\s*(\d+(\.\d+)?)", content)
        if judge_score_match:
            judge_score = float(judge_score_match.group(1))
        else:
            raise ValueError("No score found in response")
        
        return {
            'judge_message': judge_message,
            'judge_score': judge_score
        }
    except Exception as e:
        print("Error. Retrying after 10 sec", e)
        time.sleep(10)
        return create_answers(model_output, is_multi_turn)

def process_item(_, row):
    row = row[1]
    
    query_single = create_answers(row)
    query_multi = create_answers(row, is_multi_turn=True)

    row['query_single'] = query_single
    row['query_multi'] = query_multi
    row = row.to_dict()

    
    with lock:
        with open(f'judge_{time_start}.jsonl', 'a', encoding='utf-8-sig') as f:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write('\n')

with ThreadPoolExecutor(max_workers=args.threads) as executor:
    list(executor.map(process_item, df_model_outputs.index, df_model_outputs.iterrows()))
