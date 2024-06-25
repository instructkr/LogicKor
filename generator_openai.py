import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


MAX_MODEL_LEN = 1600
# MODEL = "solar-1-mini-chat"
# MODEL = "gpt4-turbo-0409"
MODEL = "gpt-4-turbo-2024-04-09"

client = OpenAI(api_key="...")

df_questions = pd.read_json("questions.jsonl", orient="records", encoding="utf-8-sig", lines=True)


def format_single_turn_question(question):
    return question[0]


single_turn_questions = df_questions["questions"].map(format_single_turn_question)
single_turn_outputs = []

for question in tqdm(single_turn_questions, desc="Processing Single Turn Questions"):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=MAX_MODEL_LEN,
        n=1,
        stop=None,
        temperature=0,
    )
    time.sleep(10)
    single_turn_outputs.append(response.choices[0].message.content.strip())


def format_double_turn_question(question, single_turn_output):
    return [question[0], single_turn_output, question[1]]


multi_turn_questions = df_questions[["questions", "id"]].apply(
    lambda x: format_double_turn_question(x["questions"], single_turn_outputs[x["id"] - 1]), axis=1
)
multi_turn_outputs = []
for question in tqdm(multi_turn_questions, desc="Processing Multi Turn Questions"):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": question[0]},
            {"role": "assistant", "content": question[1]},
            {"role": "user", "content": question[2]},
        ],
        max_tokens=MAX_MODEL_LEN,
        n=1,
        stop=None,
        temperature=0,
    )
    time.sleep(10)
    multi_turn_outputs.append(response.choices[0].message.content.strip())

df_output = pd.DataFrame(
    {
        "id": df_questions["id"],
        "category": df_questions["category"],
        "questions": df_questions["questions"],
        "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
        "references": df_questions["references"],
    }
)
df_output.to_json(f'{str(MODEL).replace("/", "_")}.jsonl', orient="records", lines=True, force_ascii=False)
