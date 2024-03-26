# LogicKor
한국어 언어모델 다분야 사고력 벤치마크

## Repository
본 Repo는 LogicKor 벤치마크의 추론 및 평가 코드, 데이터셋을 담고 있습니다.

## Evalutation Example
EEVE 템플릿, GPU 0,1 사용, model_len 4096
```
python generator.py --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --template templates/template-EEVE.json --gpu_devices 0,1 --model_len 4096
python judgement.py --model-output yanolja_EEVE-Korean-Instruct-10.8B-v1.0.jsonl --openai-api-key sk-somethingsomething --threads 30
```
