import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Union

import pandas as pd
from openai import AzureOpenAI, OpenAI

from templates import JUDGE_TEMPLATE

# Constants
TIME_START = datetime.now().strftime("%Y%m%d_%H%M%S")
LOCK = Lock()

AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", None)
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", None)
USE_AZURE_OPENAI = AZURE_ENDPOINT is not None and AZURE_DEPLOYMENT_NAME is not None and AZURE_API_VERSION is not None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--model-output-dir", help="Model Output Directory", required=True)
    parser.add_argument("-k", "--openai-api-key", help="OpenAI API Key", required=True)
    parser.add_argument("-j", "--judge-model", help="Judge Model", default="gpt-4-1106-preview")
    parser.add_argument("-t", "--threads", help="Thread count", default=42, type=int)
    parser.add_argument("--azure", help="Use Azure OpenAI", action="store_true")
    return parser.parse_args()


def create_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


def create_azure_openai_client(api_key: str):
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
        api_version=AZURE_API_VERSION,
    )


def create_answers(
    client, model_output, judge_model, is_multi_turn: bool = False, i=0
) -> Dict[str, Union[str, float]]:
    model_questions = model_output["questions"]
    model_outputs = model_output["outputs"]
    model_references = model_output["references"]

    prompt = (
        f"아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. 특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
        f"**Question**\n{model_questions[0]}"
    )

    if model_references and model_references[0]:
        prompt += f"\n\n**Additional Reference**\n{model_references[0]}"

    prompt += f"\n\n**Model's Response**\n{model_outputs[0]}"

    if is_multi_turn:
        prompt += f"\n\n**Follow-up Question.**\n{model_questions[1]}"
        if model_references and model_references[1]:
            prompt += f"\n\n**Additional Reference**\n{model_references[1]}"
        prompt += f"\n\n**Model's Response**\n{model_outputs[1]}"

    prompt += "\n\n[[대화 종료. 평가 시작.]]"

    try:
        if USE_AZURE_OPENAI:  # azure
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                temperature=0.0,
                n=1,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_TEMPLATE["multi_turn" if is_multi_turn else "single_turn"],
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        else:  # openai api
            response = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                n=1,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_TEMPLATE["multi_turn" if is_multi_turn else "single_turn"],
                    },
                    {"role": "user", "content": prompt},
                ],
            )

        content = response.choices[0].message.content
        judge_message_match = re.search(r"평가:(.*?)점수:", content.replace("*", ""), re.DOTALL)
        judge_message = judge_message_match.group(1).strip() if judge_message_match else "No judge message found"
        judge_score_match = re.search(r"점수:\s*(\d+(\.\d+)?)", content.replace("*", ""))
        if judge_score_match:
            judge_score = float(judge_score_match.group(1))
        else:
            raise ValueError("No score found in response")

        return {"judge_message": judge_message, "judge_score": judge_score}

    except Exception as e:
        print("Error. Retrying after 20 sec", e)
        time.sleep(20)

        # 꼭 아래 이유가 아닐 수 있음. 핸들링 필요.
        if i > 3:
            print("Impossible prompt, aborting..!")
            return {
                "judge_message": "Impossible to judge due to repetition.",
                "judge_score": 0.0,
            }
        i += 1
        return create_answers(client, model_output, judge_model, is_multi_turn, i)


def process_item(client, row, judge_model, output_file):
    query_single = create_answers(client, row, judge_model)
    query_multi = create_answers(client, row, judge_model, is_multi_turn=True)

    row["query_single"] = query_single
    row["query_multi"] = query_multi
    row = row.to_dict()

    with LOCK:
        with output_file.open("a", encoding="utf-8-sig") as f:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def process_file(client, file_path: Path, output_dir: Path, judge_model, threads: int, args):
    print(f"- 현재 Processing : {file_path}")
    df_model_outputs = pd.read_json(file_path, lines=True)

    output_file = output_dir / file_path.relative_to(args.model_output_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for row in df_model_outputs.iterrows():
            executor.submit(process_item, client, row[1], judge_model, output_file)


def is_hidden(filepath: Path) -> bool:
    return any(part.startswith(".") for part in filepath.parts)


def main():
    args = get_args()
    if args.azure:
        client = create_azure_openai_client(args.openai_api_key)
    else:
        client = create_openai_client(args.openai_api_key)

    input_dir = Path(args.model_output_dir)
    output_dir = Path("./evaluated")

    # Filter out hidden files
    json_files = [file for file in input_dir.rglob("*.jsonl") if not is_hidden(file)]
    print(f"Found {len(json_files)} JSON files to process")

    for file_path in json_files:
        output_file_path = output_dir / file_path.relative_to(input_dir)
        if output_file_path.exists():
            print(f"이미 평가 완료.. : {file_path}")
            continue
        process_file(client, file_path, output_dir, args.judge_model, args.threads, args)
        time.sleep(20)  # to handle ratelimit!


if __name__ == "__main__":
    main()
