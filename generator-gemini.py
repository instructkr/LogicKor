import argparse  # noqa: I001
import os

import google.generativeai as genai
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from templates import PROMPT_STRATEGY

# Constants
API_KEY = "..."
MODEL_NAME = "gemini-1.5-pro-001"

# Configure the Gemini API
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Safety settings
safety_settings = {
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", help="Directory to save outputs", default="./generated")
args = parser.parse_args()

print(f"Args - {args}")

df_questions = pd.read_json("questions.jsonl", orient="records", encoding="utf-8-sig", lines=True)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(1), retry=retry_if_exception_type(Exception))
def call_gemini_api(messages):
    """Function to call the Gemini API and return the generated text."""
    response = model.generate_content(messages, safety_settings=safety_settings)

    if not response.candidates:
        raise ValueError("Invalid operation: No candidates returned in the response.")

    candidate = response.candidates[0]
    if not candidate.messages:
        print(candidate)
        raise ValueError("Invalid operation: No messages found in the candidate.")

    return candidate.messages[-1].content


for strategy_name, prompts in PROMPT_STRATEGY.items():

    def format_single_turn_question(question):
        # Make a deep copy of the prompts to avoid modifying the original
        formatted_prompts = [dict(p) for p in prompts]
        formatted_prompts.append({"role": "user", "content": question[0]})
        return formatted_prompts

    single_turn_questions = df_questions["questions"].map(format_single_turn_question)
    single_turn_outputs = []
    for messages in tqdm(single_turn_questions, desc=f"Generating single-turn outputs for {strategy_name}"):
        generated_text = call_gemini_api(messages)
        single_turn_outputs.append(generated_text)

    def format_double_turn_question(question, single_turn_output):
        # Make a deep copy of the prompts to avoid modifying the original
        formatted_prompts = [dict(p) for p in prompts]
        formatted_prompts.extend(
            [
                {"role": "user", "content": question[0]},
                {"role": "assistant", "content": single_turn_output},
                {"role": "user", "content": question[1]},
            ]
        )
        return formatted_prompts

    multi_turn_questions = df_questions[["questions", "id"]].apply(
        lambda x: format_double_turn_question(x["questions"], single_turn_outputs[x["id"] - 1]),
        axis=1,
    )
    multi_turn_outputs = []
    for messages in tqdm(multi_turn_questions, desc=f"Generating multi-turn outputs for {strategy_name}"):
        generated_text = call_gemini_api(messages)
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
