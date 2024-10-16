# OCR 함수들
import requests
from langchain_core.messages import HumanMessage
from flask import Flask, request, jsonify
import re
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage



def extract_clauses_as_dict(ocr_text):
    clauses = re.split(r'(제\s*\d+\s*조)', ocr_text)

    merged_clauses = {}
    current_clause = ""
    current_clause_number = 0

    for i in range(1, len(clauses) - 1, 2):
        clause_title = clauses[i]
        clause_content = clauses[i + 1]

        clause_number = int(re.search(r'\d+', clause_title).group())

        if current_clause_number >= clause_number:
            current_clause += clause_title + clause_content
        else:
            if current_clause:
                cleaned_text = re.sub(r'\s+', ' ', current_clause).strip()
                merged_clauses[current_clause_number] = cleaned_text

            current_clause = clause_title + clause_content
            current_clause_number = clause_number

    if current_clause:
        cleaned_text = re.sub(r'\s+', ' ', current_clause).strip()
        merged_clauses[current_clause_number] = cleaned_text

    return merged_clauses