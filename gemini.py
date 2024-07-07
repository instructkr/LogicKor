import google.generativeai as genai # noqa: I001
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

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

# Load questions
df_questions = pd.read_json("questions.jsonl", lines=True)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(1), retry=retry_if_exception_type(Exception))
def call_gemini_api(input_text):
    """Function to call the Gemini API and return the generated text."""
    response = model.generate_content(
        [input_text],
        safety_settings=safety_settings,
    )

    if not response.candidates:
        raise ValueError("Invalid operation: No candidates returned in the response.")

    candidate = response.candidates[0]
    if not candidate.content.parts:
        print(candidate)
        raise ValueError("Invalid operation: No parts found in the candidate.")

    return candidate.content.parts[0].text


# Generate single-turn outputs
single_turn_outputs = []
for question in tqdm(df_questions["questions"].map(lambda x: x[0]), desc="Generating single-turn outputs"):
    generated_text = call_gemini_api(question)
    single_turn_outputs.append(generated_text)

# Generate multi-turn outputs
multi_turn_outputs = []
for idx, row in tqdm(df_questions.iterrows(), total=df_questions.shape[0], desc="Generating multi-turn outputs"):
    question_format = f"{row['questions'][0]} {single_turn_outputs[idx]} {row['questions'][1]}"
    generated_text = call_gemini_api(question_format)
    multi_turn_outputs.append(generated_text)

# Save outputs
df_output = pd.DataFrame(
    {
        "id": df_questions["id"],
        "category": df_questions["category"],
        "questions": df_questions["questions"],
        "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
        "references": df_questions["references"],
    }
)

df_output.to_json("gemini_pro_outputs.jsonl", orient="records", lines=True, force_ascii=False)
