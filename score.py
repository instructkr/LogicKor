import argparse
import glob

import pandas as pd

# 파일 경로 패턴
# file_pattern = './judge_20240418_103542.jsonl'
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--print", help="judge Output File Location", default=None)
args = parser.parse_args()

if args.print is None:
    raise ValueError("Judge Output File Location is required")

# 카테고리별 점수 집계를 위한 딕셔너리
category_scores = {}

# 전체 싱글 점수와 멀티 점수의 리스트
total_single_scores = []
total_multi_scores = []

# 지정된 패턴에 맞는 모든 파일을 찾아서 처리
for file_path in glob.glob(args.print):
    file = pd.read_json(file_path, orient="records", encoding="utf-8-sig", lines=True)
    for item in file.to_dict(orient="records"):
        category = item["category"]
        single_score = item["query_single"]["judge_score"]
        multi_score = item["query_multi"]["judge_score"]

        if category not in category_scores:
            category_scores[category] = {"single_scores": [], "multi_scores": []}

        category_scores[category]["single_scores"].append(single_score)
        category_scores[category]["multi_scores"].append(multi_score)

        # 전체 점수 리스트에 추가
        total_single_scores.append(single_score)
        total_multi_scores.append(multi_score)

# 표의 헤더 생성
table_header = "| Category | Single turn | Multi turn |\n|---|---|---|"

# 표의 내용 생성
table_rows = []
for category, scores in category_scores.items():
    avg_single = sum(scores["single_scores"]) / len(scores["single_scores"])
    avg_multi = sum(scores["multi_scores"]) / len(scores["multi_scores"])
    table_rows.append(f"| {category} | {avg_single:.2f} | {avg_multi:.2f} |")

    total_single_scores.extend(scores["single_scores"])
    total_multi_scores.extend(scores["multi_scores"])

# 카테고리별 점수 평균 출력
print(table_header)
for row in table_rows:
    print(row)

# 전체 점수의 평균 계산 및 출력
avg_total_single = sum(total_single_scores) / len(total_single_scores)
avg_total_multi = sum(total_multi_scores) / len(total_multi_scores)
avg_total = (avg_total_single + avg_total_multi) / 2

# 전체 점수 평균 출력
print("\n| Category | Score |\n|---|---|")
print(f"| Single turn | {avg_total_single:.2f} |")
print(f"| Multi turn | {avg_total_multi:.2f} |")
print(f"| Overall | {avg_total:.2f} |")
