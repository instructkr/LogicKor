import argparse
import os

import google.generativeai as genai
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from templates import PROMPT_STRATEGY

# TODO: generator-gemini.py to converge with generator.py
API_KEY = "..."
MODEL_NAME = "gemini-1.5-pro-001"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

safety_settings = {
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", help="Directory to save outputs", default="./generated")
args = parser.parse_args()

df_questions = pd.read_json("questions.jsonl", orient="records", encoding="utf-8-sig", lines=True)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(1), retry=retry_if_exception_type(Exception))
def call_gemini_api(input_text):
    """Function to call the Gemini API and return the generated text."""
    response = model.generate_content([input_text], safety_settings=safety_settings)

    if not response.candidates:
        raise ValueError("Invalid operation: No candidates returned in the response.")

    candidate = response.candidates[0]
    if not candidate.content.parts:
        print(candidate)
        raise ValueError("Invalid operation: No parts found in the candidate.")

    return candidate.content.parts[0].text


for strategy_name, prompts in PROMPT_STRATEGY.items():

    def format_single_turn_question(question):
        messages = prompts + [{"role": "user", "content": question[0]}]
        formatted_text = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
        return formatted_text

    single_turn_questions = df_questions["questions"].map(format_single_turn_question)
    single_turn_outputs = []
    for formatted_text in tqdm(single_turn_questions, desc=f"Generating single-turn outputs for {strategy_name}"):
        generated_text = call_gemini_api(formatted_text)
        single_turn_outputs.append(generated_text)

    def format_double_turn_question(question, single_turn_output):
        messages = prompts + [
            {"role": "user", "content": question[0]},
            {"role": "assistant", "content": single_turn_output},
            {"role": "user", "content": question[1]},
        ]
        formatted_text = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
        return formatted_text

    multi_turn_questions = df_questions[["questions", "id"]].apply(
        lambda x: format_double_turn_question(x["questions"], single_turn_outputs[x["id"] - 1]),
        axis=1,
    )
    multi_turn_outputs = []
    for formatted_text in tqdm(multi_turn_questions, desc=f"Generating multi-turn outputs for {strategy_name}"):
        generated_text = call_gemini_api(formatted_text)
        multi_turn_outputs.append(generated_text)

    df_output = pd.DataFrame(
        {
            "id": df_questions["id"],
            "category": df_questions["category"],
            "questions": df_questions["questions"],
            "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
            "references": df_questions["references"],
        }
    )
    output_path = os.path.join(args.output_dir, f"{strategy_name}.jsonl")
    df_output.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved outputs to {output_path}")
