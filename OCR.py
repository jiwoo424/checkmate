# OCR 함수들
import requests
from langchain_core.messages import HumanMessage
from flask import Flask, request, jsonify
import re
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage



# OCR 텍스트에서 '제 x 조' 패턴으로 시작하고, 문장부호로 끝나는 조항을 인식하여 따로 저장하는 함수
def extract_clauses_with_order(ocr_text):
    clauses = []
    processed_text = []

    # "제 x 조" 패턴으로 시작하고, 마침표로 끝나는 조항을 모두 인식
    clause_pattern = re.compile(r'(제\s?\d+\s?조[^.!?]*[.!?])', re.DOTALL)  # 조항이 중간에 줄바꿈이 있어도 인식되도록 DOTALL 플래그 사용
    matches = clause_pattern.finditer(ocr_text)

    last_index = 0
    for match in matches:
        start, end = match.span()
        # 조항 전의 텍스트를 처리
        pre_text = ocr_text[last_index:start].strip()
        if pre_text:
            processed_text.append(pre_text)
        # 조항을 인식하고 저장
        clauses.append(match.group().strip())
        processed_text.append(f"조항: {match.group().strip()}")
        last_index = end

    # 남은 텍스트 처리
    remaining_text = ocr_text[last_index:].strip()
    if remaining_text:
        processed_text.append(remaining_text)

    return clauses, processed_text

# 불필요한 줄바꿈을 제거하는 함수
def clean_text(text):
    return text.replace('\n', ' ').strip()

# 나머지 텍스트에 대해서 조항이 아닌 부분은 전부 '해당없음'으로 처리하는 함수
def classify_remaining_text(remaining_text_list):
    results = []

    # 남은 텍스트를 줄바꿈 기준으로 나누어 처리
    for line in remaining_text_list:
        if line.startswith("조항:"):
            results.append(line)  # 조항은 그대로 처리
        else:
            # 남은 텍스트의 각 줄에 대해 '해당없음' 처리
            lines = line.split("\n")
            for l in lines:
                l = l.strip()
                if l:
                    results.append(f"해당없음: {clean_text(l)}")

    return results

# 최종적으로 조항을 분리하고 나머지를 처리하는 함수, 결과를 딕셔너리로 저장
def process_ocr_text(ocr_text):
    # 1. 조항을 추출하고 순서를 유지하여 텍스트를 처리
    clauses, remaining_text_list = extract_clauses_with_order(ocr_text)

    # 2. 나머지 텍스트에 대해 전부 '해당없음' 처리
    classified_remaining_text = classify_remaining_text(remaining_text_list)

    # 3. 딕셔너리에 저장 (조항과 해당없음)
    final_dict = {}
    for idx, entry in enumerate(classified_remaining_text):
        if entry.startswith("조항:"):
            final_dict[f"item_{idx+1}"] = {"type": "조항", "content": entry.split(": ", 1)[1]}
        else:
            final_dict[f"item_{idx+1}"] = {"type": "해당없음", "content": entry.split(": ", 1)[1]}

    return final_dict