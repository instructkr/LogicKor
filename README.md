# LogicKor
한국어 언어모델 다분야 사고력 벤치마크

## Benchmark Website
https://lk.instruct.kr/

## Note
pr 적극 환영합니다.
벤치마크 결과 Self-Report도 받습니다. issue나 pr 부탁드립니다. 💕

## Repository
본 Repo는 LogicKor 벤치마크의 추론 및 평가 코드, 데이터셋을 담고 있습니다.

## Evalutation Example
EEVE 템플릿, GPU 0,1 사용, model_len 4096
```
python generator.py --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --gpu_devices 0,1 --model_len 4096
python judgement.py -o yanolja_EEVE-Korean-Instruct-10.8B-v1.0.jsonl -k sk-somethingsomething -t 30
python score.py -p ./results/judge_HyperClovaX.jsonl 
```

vllm으로 실행해 API로 평가하고자 하시는 경우 `generator-vllm.py` 를 이용할 수 있으며 vllm으로 서빙할 때 `--chat-template` 을 통해 토크나이저의 기본 config를 명시하는 것을 추천합니다.

```
python ./entrypoints/openai/api_server.py  --dtype half --model Qwen/Qwen2-72B-Instruct --served-model-name ... --chat-template "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```
