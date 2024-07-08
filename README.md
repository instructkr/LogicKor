# LogicKor

한국어 언어모델 다분야 사고력 벤치마크

## Benchmark Website

<https://lk.instruct.kr>

## Note

pr 적극 환영합니다.
벤치마크 결과 Self-Report도 받습니다. issue나 pr 부탁드립니다. 💕
* 권장 사항: PR 이전에 `make format && make check` 를 통해 코드 포맷팅을 확인해주세요. (black, isort, ruff 의존성 설치 필요)

## Repository

본 Repo는 LogicKor 벤치마크의 추론 및 평가 코드, 데이터셋을 담고 있습니다.

## Evaluation Example

EEVE 템플릿, GPU 0,1 사용, model_len 4096

### 1. 인퍼런스 결과 생성

```bash
python generator.py --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --gpu_devices 0,1 --model_len 4096
```

### 2. Judge 모델로 평가

#### OpenAI

```bash
python evaluator.py -o ./generated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0 -k sk-somethingsomething -t 30
```

#### Azure

```bash
export AZURE_ENDPOINT=$AZURE_ENDPOINT
export AZURE_DEPLOYMENT_NAME=$AZURE_DEPLOYMENT_NAME
export AZURE_API_VERSION=$AZURE_API_VERSION

python evaluator.py --azure -o ./generated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0 -k sk-somethingsomething -t 30
```

### 3. 결과 확인

```bash
python score.py -p ./evaluated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/default.jsonl
```
