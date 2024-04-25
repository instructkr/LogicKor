import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_devices', help=' : CUDA_VISIBLE_DEVICES', default='0')
parser.add_argument('--model', help=' : Model to evaluate', default='yanolja/EEVE-Korean-Instruct-2.8B-v1.0')
parser.add_argument('--template', help=' : Template File Location', default='./templates/template-EEVE.json')
parser.add_argument('--model_len', help=' : Maximum Model Length', default=4096, type=int)
parser.add_argument('--batch_size', help=' : Batch Size', default=8, type=int)
parser.add_argument('--num_workers', help=' : Number of DataLoader Workers', default=4, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
gpu_counts = len(args.gpu_devices.split(','))

df_config = pd.read_json(args.template, typ='series')
SINGLE_TURN_TEMPLATE = df_config.iloc[0]
MULTI_TURN_TEMPLATE = df_config.iloc[1]

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


class QuestionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


def collate_fn(batch):
    return pd.DataFrame(batch)


def process_batch(batch: pd.DataFrame):
    _single_turn_questions = batch['questions'].apply(lambda x: format_single_turn_question(x))
    # Convert tp `list[str]` for `llm.generate`
    single_turn_questions = _single_turn_questions.tolist() if isinstance(_single_turn_questions, pd.Series) else _single_turn_questions

    single_turn_outputs = [
        output.outputs[0].text.strip()
        for output in llm.generate(single_turn_questions, sampling_params)
    ]
    # Convert to `pd.Series` for multi-turn question generation
    single_turn_outputs = pd.Series(single_turn_outputs, index=batch.index)

    def format_multi_turn_question(row):
        return MULTI_TURN_TEMPLATE.format(
            row['questions'][0], single_turn_outputs[row.name], row['questions'][1]
        )

    _multi_turn_questions = batch.apply(format_multi_turn_question, axis=1)
    # Convert tp `list[str]` for `llm.generate``
    multi_turn_questions = _multi_turn_questions.tolist() if isinstance(_multi_turn_questions, pd.Series) else _multi_turn_questions

    multi_turn_outputs = [
        output.outputs[0].text.strip()
        for output in llm.generate(multi_turn_questions, sampling_params)
    ]
    multi_turn_outputs = pd.Series(multi_turn_outputs, index=batch.index)

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
        prefetch_factor=2,
        pin_memory=True
    )

    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(process_batch, dataloader))
    # Disable ThreadPoolExecutor due to conflict with `vllm` scheduler.
    results = list(map(process_batch, dataloader))

    df_output = pd.concat(results, ignore_index=True)
    df_output.to_json(
        f'{str(args.model).replace("/", "_")}.jsonl',
        orient='records',
        lines=True,
        force_ascii=False
    )


process_data(df_questions, args.batch_size, args.num_workers)
