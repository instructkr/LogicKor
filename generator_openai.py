import argparse
import pandas as pd
from openai import OpenAI
import os
import time
from tqdm import tqdm


if __name__ == '__main__':

    PROMPT_STRATEGY = {
        "cot-1-shot": [
            {"role": "system", "content": "문제를 해결하는 과정을 단계별로 명확하게 설명한 다음, 최종 답변을 제시해 주세요."},
            {"role": "user", "content": "한 도시에서 매년 나무를 심고 있습니다. 첫해에 100그루를 심었고, 매년 전년도에 심은 나무의 수보다 20%씩 더 많은 나무를 심고 있습니다. 2년 후에는 총 몇 그루의 나무를 심게 될까요?"},
            {"role": "assistant", "content": """이 문제를 해결하기 위해 단계별로 접근해 보겠습니다.

1단계: 주어진 정보 정리
- 첫해에 100그루를 심음
- 매년 전년도보다 20% 더 많은 나무를 심음
- 2년 후의 총 나무 수를 구해야 함

2단계: 두 번째 해에 심은 나무 수 계산
- 첫해보다 20% 증가
- 100 + (100의 20%) = 100 + 20 = 120그루

3단계: 세 번째 해에 심은 나무 수 계산
- 두 번째 해보다 20% 증가
- 120 + (120의 20%) = 120 + 24 = 144그루

4단계: 3년 동안 심은 총 나무 수 계산
- 첫해: 100그루
- 둘째 해: 120그루
- 셋째 해: 144그루
- 총합: 100 + 120 + 144 = 364그루

따라서, 2년 후(즉, 3년째)에는 총 364그루의 나무를 심게 됩니다."""}
        ],
        
        "1-shot": [
            {"role": "user", "content": "Git의 rebase와 merge의 차이점을 설명하고, 각각 어떤 상황에서 사용하는 것이 좋은지 예를 들어 설명해주세요."},
            {"role": "assistant", "content": """Git에서 `rebase`와 `merge`는 두 개의 브랜치를 통합하는 데 사용되는 두 가지 주요 명령어입니다. 이들의 차이점과 사용 상황을 설명하겠습니다.

### Rebase

#### 개념
- `rebase`는 하나의 브랜치의 커밋을 다른 브랜치의 끝에 다시 적용하는 것입니다. 즉, 커밋 히스토리를 새로 만든다고 할 수 있습니다.
- 결과적으로 커밋 히스토리가 깔끔하게 정리됩니다.

#### 사용 상황
- **깔끔한 히스토리가 필요할 때**: 브랜치 히스토리가 병합 없이 일직선으로 이어지도록 하고 싶을 때 유용합니다.
- **작업 중인 브랜치를 최신 상태로 유지하고 싶을 때**: 예를 들어, `feature` 브랜치에서 작업 중인데 `main` 브랜치에서 업데이트가 있을 경우, `feature` 브랜치를 `rebase`하여 최신 상태로 유지할 수 있습니다.

#### 예제
1. `feature` 브랜치에서 작업 중입니다.
2. `main` 브랜치에 새로운 커밋이 추가되었습니다.
3. `feature` 브랜치에서 `git rebase main`을 실행합니다.
4. `feature` 브랜치의 커밋들이 `main` 브랜치의 최신 커밋들 뒤에 재배치됩니다.

```bash
git checkout feature
git rebase main
```

### Merge

#### 개념
- `merge`는 두 개의 브랜치를 합치는 방법으로, 두 브랜치의 히스토리를 유지하며 새로운 병합 커밋을 생성합니다.
- `merge`는 기존의 히스토리를 그대로 보존합니다.

#### 사용 상황
- **히스토리를 유지하고 싶을 때**: 각 브랜치의 작업 기록을 그대로 보존하면서 병합할 때 사용합니다.
- **협업 중 충돌을 명확히 해결하고 기록을 남기고 싶을 때**: 여러 사람이 같은 저장소에서 작업할 때, `merge`는 충돌 해결 과정과 그 기록을 명확히 남길 수 있습니다.

#### 예제
1. `feature` 브랜치에서 작업을 완료했습니다.
2. `main` 브랜치에 병합하고 싶습니다.
3. `main` 브랜치로 체크아웃한 후 `feature` 브랜치를 병합합니다.

```bash
git checkout main
git merge feature
```

### 요약

- **Rebase**는 히스토리를 깔끔하게 정리하는 데 유리하며, 주로 개인 작업이나 `feature` 브랜치를 최신 상태로 유지할 때 사용됩니다.
- **Merge**는 두 브랜치의 작업 히스토리를 보존하면서 병합하며, 협업 과정에서 충돌 해결과 기록을 명확히 남기는 데 유리합니다.

각 방법의 장단점과 사용 상황을 고려하여 적절히 선택하는 것이 중요합니다."""}
    ],
        "default": [],
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_base', help=' : Url to connect', default='https://api.openai.com/v1')
    parser.add_argument('--api_key', help=' : Key to connect', default=' ')
    parser.add_argument('--model', help=' : Model to evaluate', default='gpt-4o')
    parser.add_argument('--model_len', help=' : Maximum Model Length', default=4096, type=int)
    args = parser.parse_args()
    
    print(args)
    
    class LLMWrapper:
        def __init__(self, client):
            self.client = client
        def generate(self, prompts):
            responses = []
            for i, prompt in enumerate(tqdm(prompts, desc="프롬프트 처리중...")):
                response = self.client.chat.completions.create(
                    model=args.model,
                    messages=prompt,
                    max_tokens=args.model_len,
                    temperature=0,
                    stop=['<|endoftext|>', '[INST]', '[/INST]', '<|im_end|>', '<|end|>'],
                )
                responses.append(response.choices[0].message.content)
            return response

    llm = LLMWrapper(OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=3600*24))
    
    df_questions = pd.read_json('questions.jsonl', lines=True)
    
    if not os.path.exists("./generated/" + args.model):
        os.makedirs("./generated/" + args.model)
    
    for strategy_name, prompts in PROMPT_STRATEGY.items():
        def format_single_turn_question(question):
            return prompts + [{"role": "user", "content": question[0]}]
        
        single_turn_questions = df_questions['questions'].map(format_single_turn_question)
        print(single_turn_questions.iloc[0])
        single_turn_outputs = [output for output in llm.generate(single_turn_questions)]
        
        def format_double_turn_question(question, single_turn_output):
            return prompts + [{"role": "user", "content": question[0]}, {"role": "assistant", "content": single_turn_output}, {"role": "user", "content": question[1]}]
        
        multi_turn_questions = df_questions[['questions', 'id']].apply(lambda x: format_double_turn_question(x['questions'], single_turn_outputs[x['id']-1]), axis=1)
        multi_turn_outputs = [output for output in llm.generate(multi_turn_questions)]
        
        df_output = pd.DataFrame({'id': df_questions['id'], 'category': df_questions['category'], 'questions': df_questions['questions'], 'outputs': list(zip(single_turn_outputs, multi_turn_outputs)), "references": df_questions['references']})
        df_output.to_json('./generated/' + os.path.join(args.model, f'{strategy_name}.jsonl'), orient='records', lines=True, force_ascii=False)