import os
import argparse
import pandas as pd
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_devices', help=' : CUDA_VISIBLE_DEVICES', default='0')
parser.add_argument('--model', help=' : Model to evaluate', default='yanolja/EEVE-Korean-Instruct-2.8B-v1.0')
parser.add_argument('--template', help=' : Template File Location', default='./templates/template-EEVE.json')
parser.add_argument('--model_len', help=' : Maximum Model Length', default=4096, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
gpu_counts = len(args.gpu_devices.split(','))

df_config = pd.read_json(args.template, typ='series')
SINGLE_TURN_TEMPLATE = df_config.iloc[0]
DOUBLE_TURN_TEMPLATE = df_config.iloc[1]

llm = LLM(
    model=args.model,
    tensor_parallel_size=gpu_counts,
    max_model_len=int(args.model_len),
    gpu_memory_utilization=0.95
)
sampling_params = SamplingParams(
    temperature=0,
    top_p=1,
    top_k=-1,
    early_stopping=True,
    best_of=4,
    use_beam_search=True,
    skip_special_tokens=False,
    max_tokens=args.model_len,
    stop=['<|endoftext|>', '</s>', '<|im_end|>', '[INST]', '[/INST]', '<end_of_turn>', '<start_of_turn>']
)

df_questions = pd.read_json('questions.jsonl', lines=True)

def format_single_turn_question(question):
    return SINGLE_TURN_TEMPLATE.format(question[0])

single_turn_questions = df_questions['questions'].map(format_single_turn_question)
single_turn_outputs = [
    output.outputs[0].text.strip()
    for output in llm.generate(single_turn_questions, sampling_params)
]

def format_double_turn_question(question, single_turn_output):
    return DOUBLE_TURN_TEMPLATE.format(
        question[0], single_turn_output, question[1]
    )

multi_turn_questions = df_questions[['questions', 'id']].apply(
    lambda x: format_double_turn_question(x['questions'], single_turn_outputs[x['id'] - 1]),
    axis=1
) # bad code ig?

multi_turn_outputs = [
    output.outputs[0].text.strip()
    for output in llm.generate(multi_turn_questions, sampling_params)
]

df_output = pd.DataFrame({
    'id': df_questions['id'],
    'category': df_questions['category'],
    'questions': df_questions['questions'],
    'outputs': list(zip(single_turn_outputs, multi_turn_outputs)),
    'references': df_questions['references']
})
df_output.to_json(
    f'{str(args.model).replace("/", "_")}.jsonl',
    orient='records',
    lines=True,
    force_ascii=False
)
