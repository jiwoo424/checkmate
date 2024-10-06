from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import streamlit as st
from langchain.llms.base import LLM
from typing import Optional



def extract_legal_terms(clause, terms_df):
    terms_in_clause = []
    for term in terms_df['term']:
        if term in clause:
            terms_in_clause.append(term)
    return terms_in_clause

def legal_explanations(terms, terms_df):
    explanations = {}
    for term in terms:
        explanation = terms_df[terms_df['term'] == term]['definition'].values
        if explanation:
            explanations[term] = explanation[0]
    return explanations

terms_df = pd.read_csv("web_terms.csv")
preceds_df = pd.read_csv("판례.csv")
clauses_df = pd.read_csv("조항.csv")

# 법률 용어 설명
def legal_explanations(terms, terms_df):
    explanations = {}
    for term in terms:
        explanation = terms_df[terms_df['term'] == term]['definition'].values
        if explanation:
            explanations[term] = explanation[0]
    return explanations

api_key = st.secrets['API_KEY']


class ChatUpstageLLM(LLM):
    def __init__(self, model: str, upstage_api_key: str):
        self.chat_upstage = ChatUpstage(model=model, upstage_api_key=api_key)

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        # ChatUpstage의 generate 또는 비슷한 메서드를 사용하여 응답을 생성
        response = self.chat_upstage.generate(prompt)
        return response.content

    @property
    def _llm_type(self) -> str:
        return "upstage"


def generate_clause_explanation(clause, term_explanations, detection=False, corr_ex=None, judgment=None):
    # Upstage 모델 초기화
    model = 'solar-1-mini-chat'
    llm = ChatUpstage(model=model, upstage_api_key=api_key)


    # LangChain 프롬프트 템플릿 설정
    if not detection:
        explanation_template = """
        주어진 조항: "{clause}"

        용어 설명: {term_explanations}

        용어 설명을 이용해서, 주어진 조항을 일반인도 이해하기 쉽도록 설명해.
        """
        explanation_prompt = PromptTemplate(template=explanation_template, input_variables=["clause", "term_explanations"])
    else:
        explanation_template = """
        주어진 조항: "{clause}"

        용어 설명: {term_explanations}

        유사 과거 조항: "{corr_ex}"

        유사 과거 조항에 대한 해석: {judgment}

        주어진 조항은 유사 과거 조항으로 감지된 조항이다.
        유사 과거 조항에 대한 해석을 바탕으로 주어진 조항이 불합리한 이유를 쉽게 설명해.
        """
        explanation_prompt = PromptTemplate(template=explanation_template, input_variables=["clause", "term_explanations", "corr_ex", "judgment"])

    # LLMChain을 사용하여 프롬프트와 LLM을 연결

    chain = explanation_prompt|llm

    # 조항 설명 생성
    if not detection:
        simplified_clause = chain.run({"clause": clause, "term_explanations": term_explanations})
    else:
        simplified_clause = chain.run({"clause": clause, "term_explanations": term_explanations, "corr_ex": corr_ex, "judgment": judgment})

    return simplified_clause