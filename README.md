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
python generator.py --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --template templates/template-EEVE.json --gpu_devices 0,1 --model_len 4096
python judgement.py --model-output yanolja_EEVE-Korean-Instruct-10.8B-v1.0.jsonl --openai-api-key sk-somethingsomething --threads 30
```
