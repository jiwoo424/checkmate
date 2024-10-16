# OCR 함수들
import requests
from langchain_core.messages import HumanMessage
from flask import Flask, request, jsonify
import re
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage



def extract_clauses_as_dict(ocr_text):
    # 조항을 "제 ~ 조" 패턴으로 분리, 조항 번호를 포함하여 분리
    clauses = re.split(r'(제\s*\d+\s*조)', ocr_text)

    # 조항 딕셔너리 생성
    merged_clauses = {}
    current_clause = ""
    current_clause_number = 0

    for i in range(1, len(clauses) - 1, 2):
        # '제 ~ 조'로 시작하는 부분을 현재 조항의 제목으로 설정
        clause_title = clauses[i]
        clause_content = clauses[i + 1]

        # 현재 조항의 번호를 추출
        clause_number = int(re.search(r'\d+', clause_title).group())

        # 언급된 조항이 현재 조항 번호보다 작거나 같을 경우 포함
        if current_clause_number >= clause_number:
            current_clause += clause_title + clause_content
        else:
            # 불필요한 줄바꿈과 공백 제거
            if current_clause:
                # 기본 공백과 줄바꿈 제거
                cleaned_text = re.sub(r'\s+', ' ', current_clause).strip()
                merged_clauses[current_clause_number] = cleaned_text

            # 다음 조항으로 이동
            current_clause = clause_title + clause_content
            current_clause_number = clause_number

    # 마지막 조항 추가
    if current_clause:
        cleaned_text = re.sub(r'\s+', ' ', current_clause).strip()
        merged_clauses[current_clause_number] = cleaned_text

    return merged_clauses