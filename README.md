# LogicKor

한국어 언어모델 다분야 사고력 벤치마크

## 🚧 LogicKor 운영 관련 공지 🚧
안녕하세요. LogicKor 깃허브 Repository를 Read-Only로 전환 및 리더보드 업데이트 중지 계획에 대하여 말씀드리고자 합니다.

최근 출시되는 모델들의 성능이 점차 상향 평준화 되어가면서 리더보드에서의 상위권 모델에 대한 변별력이 거의 없어졌습니다. 이에 따라 데이터셋이나 평가 방식의 변경이 필요하지만 라이브로 운영되는 리더보드 특성상 진행하기 힘든 작업입니다.
또한, 리더보드에 주기적으로 신규 모델을 추가하고 있지 못하는 등의 운영적인 측면에서 충분한 관리를 하지 못하고 있습니다. 주요 Contributor들이 시간을 들이기에 어려운 상황입니다.
처음 리더보드 운영의 목적이었던 '실제로 느껴지는 대로의 점수를 제공하자'라는 취지에 반하는 모델이 생겨나며 원래의 목적성을 잃기도 하였습니다.

이러한 이유로 LogicKor Repo를 Read-Only로 전환하고 리더보드는 현재 상태를 유지하는 결정을 내리게 되었습니다.
많이 관심 가져주시고 이용해주셔서 감사했습니다. 추후, 제기한 여럿 문제점들을 개선한 LogicKor Hard로 돌아오겠습니다.

## Benchmark Website

<https://lk.instruct.kr>

## Note

pr 적극 환영합니다.
벤치마크 결과 Self-Report도 받습니다. issue나 pr 부탁드립니다. 💕
* 권장 사항: PR 이전에 `make format && make check` 를 통해 코드 포맷팅을 확인해주세요. (black, isort, ruff 의존성 설치 필요)

## Repository

본 Repo는 LogicKor 벤치마크의 추론 및 평가 코드, 데이터셋을 담고 있습니다.

## Evaluation Example

GPU 0,1 사용, model_len 4096

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
