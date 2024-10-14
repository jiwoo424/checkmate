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
from langchain_community.vectorstores import Chroma

from DETECTION import initialize_embeddings, load_vector_store

import wikipedia
from tavily import TavilyClient


persist_directory = "./chroma_data"
persist_directory_db = "./chroma_db"


api_key = st.secrets['API_KEY']
embeddings = initialize_embeddings(api_key)
vector_store = load_vector_store(persist_directory, embeddings)
db = load_vector_store(persist_directory_db, embeddings)
retriever = db.as_retriever()


def extract_legal_terms(clause, terms_df):
    terms_in_clause = []
    for term in terms_df['term']:
        if term in clause:
            terms_in_clause.append(term)
    return terms_in_clause


terms_df = pd.read_csv("web_terms.csv")
preceds_df = pd.read_csv("판례.csv")
clauses_df = pd.read_csv("조항.csv")

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
        response = self.chat_upstage.generate(prompt)
        return response.content

    @property
    def _llm_type(self) -> str:
        return "upstage"


def generate_clause_explanation(clause, term_explanations, detection=False, corr_ex=None, judgment=None):
    model = 'solar-1-mini-chat'
    llm = ChatUpstage(model=model, upstage_api_key=api_key)

    if not detection:
      explanation_template = """
    주어진 조항: "{clause}"

    용어 설명: {term_explanations}

    용어 설명을 이용해서, 주어진 조항을 일반인도 이해하기 쉽도록 설명해.
    """
      explanation_prompt = PromptTemplate(template=explanation_template, input_variables=["clause", "term_explanations"])
    else:
      explanation_template = """
    주어진 조항은 불리한 조항으로 감지된 조항이다.
    유사 과거 조항에 대한 해석을 바탕으로 주어진 조항이 불합리한 이유를 쉽게 설명해.

    <예시1>
    주어진 조항: "제10항\n입주자는 계약기간을 종료하기전 다음 세입자를 선정해서 임대를 인계해야하고 다음 입주자의 임대보증금을 회사에 입금한 후 환불받는다."

    용어 설명: '계': '상부 상조 친목 공동이익 등을 목적으로 만들어진 한국의 전통 협동조직으로 모임',
    '계약': '법률효과 발생을 목적으로 한 두 개의 의사표시가 합치함으로써 성립하는 하나의 법률행위이다',
    '기간': '일정한 시점에서 다른 시점까지의 시간적인 간격을 의미한다 기간은 그것만으로는 법률요건이 성립되지 않으나 기간의 만료에 의하여 중요한 법률효과를 발생시키는 경우가 많다',
    '보증금': '보증금이란 미래에 발생할 수 있는 서로간의 불이익을 막고자 임차인이 미리 임대인에게 지급하는 금전을 의미합니다 예를 들면 전세계약의 전세금은 보증금에 해당합니다',
    '임대': '임대란 계약의 당사자 가운데 한쪽이 상대편에게 부동산 등 물건을 사용하게 하고 상대편은 이에 대하여 일정한 임차료를 지급할 것을 약속하는 계약입니다 계약을 통해서 빌려 주는 사람은 임대인이 되고 빌리는 사람은 임차인이 됩니다',
    '세입자': '세입자는 일정한 세를 내고 남의 건물이나 방 따위를 빌려 쓰는 사람을 말합니다'

    유사 과거 조항: "제2항\n입주자는 계약기간을 종료하기전 다음 세입자를 선정하여 임대를 인계하고 다음 입주자의 임대보증금을 회사에 입금후 환불받는다."

    유사 과거 조항에 대한 해석: 해당 약관조항은 법률의 규정에 의한 고객의 해지권을 배제하거나 그 행사를 제한하는 조항이며, 계약의 해지로 인한 고객의 원상회복의무를 상당한 이유없이 과중하게 부담시키거나 원상회복청구권을 부당하게 포기하도록 하는 조항이다.

    답변: 주택의 입주자는 본 계약이 끝나기 끝나기 전에, 새로운 세입자를 구해서 그 사람에게 주택을 넘겨야 합니다. 새로운 사람이 보증금을 회사에 보내면 자신이 냈던 보증금을 돌려 받는다는 뜻입니다.
    즉 명시된 계약기간이 끝나기 전에 새로운 계약자를 구해야 하고, 보증금을 돌려받는 시점은 새로운 계약자가 보증금을 지불 한 이후라는 점입니다. 이는 세입자 입장에서 불합리한 조항으로 적용될 수 있으며, 여태까지의 인수인계 과정에 차질이 없었는지 확인해 볼 필요가 있습니다.

    <예시2>
    주어진 조항: "제8조(임대차 등기 등)\n제1항 임차인은 주택임대차보호법에 따라 임대주택에 관한 대항력을 갖추기로 한다. 그리고 갑에게 임대주택에 대한 임차권등기, 전세권등기 또는 (근)저당권등기 등을 요구할 수 없다."

    용어 설명: '대항력': '이미 유효하게 발생하고 있는 법률관계를 제자에 대하여 주장할 수 있는 법률상의 효력을 말한다',
    '임대차': '임대인이 임차인에게 어떤 물건을 사용수익하게 할 것을 약정하고 임차인이 이에 대하여 차임을 지급할 것을 약정함으로써 성립하는 계약민법 제조제조',
    '저당권': '채권자가 채무자나 또는 제자가 채무담보로서 제공한 부동산 또는 부동산물권을 인도받지 않고 다만 관념상으로만 지배하여 채무의 변제가 없을 경우 그 목적물로부터 우선변제를 받을 수 있는 권리',
    '전세권': '전세금을 지급하고 타인의 부동산을 일정기간 그 용도에 따라 사용 수익한 후 그 부동산을 반환하고 전세금을 다시 돌려받는 권리민법 제조',
    '등기': '등기란 법정절차에 따라서 부동산의 권리관계를 등기부에 등록하는 행위 또는 기재 그 자체를 의미합니다',
    '임대': '임대란 계약의 당사자 가운데 한쪽이 상대편에게 부동산 등 물건을 사용하게 하고 상대편은 이에 대하여 일정한 임차료를 지급할 것을 약속하는 계약입니다 계약을 통해서 빌려 주는 사람은 임대인이 되고 빌리는 사람은 임차인이 됩니다',
    '임대주택': '임대주택이란 국가 또는 민간 건설업체가 건축하여 주민에게 임대하는 주택입니다',
    '임차': '임차란 돈을 내고 타인의 건물을 빌리는 것을 의미합니다',
    '임차권': '임차권이란 임대차 계약에서 빌려 쓰는 사람이 그 건물을 사용하고 이익을 얻을 수 있는 권리를 의미합니다',
    '임차권등기': '임차권등기란 임대차 계약이 종료됐으나 보증금을 돌려 받지 못한 상태에서 이사를 가야 할 경우에 대항력을 유지하기 위해 설정하는 등기를 의미합니다',
    '임차인': '임차인이란 임대차 계약에서 돈을 내고 건물을 빌려 쓰는 사람입니다',
    '주택임대차보호법': '주택임대차보호법이란 국민 주거생활의 안정을 보장함을 목적으로 주거용 건물의 임대차에 관하여 민법에 대한 특례를 규정한 법률입니다',
    '전세': '전세란 주택이나 건물을 가진 사람에게 일정한 금액을 맡기고 그 주택이나 집을 일정 기간 동안 빌리는 것을 말합니다'

    유사 과거 조항: "제16조(임대차 등기 등)\n제1항 임차인은 주택임대차보호법에 따라 임대주택에 관한 대항력을 갖추기로 하며, 갑에게 임대주택에 관한 임차권등기, 전세권등기, 또는 (근)저당권등기를 요구할 수 없다."

    유사 과거 조항에 대한 해석: 특별한 사유도 없이 일방적으로 임차인이 임대인에게 임대주택에 관한 임대차등기 등을 요구할 수 없도록 규정하고 있는 바, 이는 민법 제621조에서 규정하고 있는 임차인의 법률상 권리를 상당한 이유 없이 배제하고 있다.

    답변: 임차인은 임대주택에 대해 주택임대차보호법에 따른 대항력을 갖추기로 되어 있지만, 본 계약에서는 임차인이 임대인에게 임차권등기, 전세권등기, 또는 (근)저당권등기를 요구할 수 없다고 명시하고 있습니다. 즉, 임차인은 특별한 사유 없이도 임대인에게 이러한 등기를 요구할 수 없다는 뜻입니다.
    이 조항은 임차인의 권리를 상당한 이유 없이 배제하는 것으로 볼 수 있습니다. 민법에서는 임차인이 필요할 때 임차권등기를 통해 자신의 권리를 보호할 수 있도록 하고 있는데, 이 조항은 그 권리를 제한하고 있기 때문입니다. 임차권등기나 전세권등기는 임차인이 주택의 소유권 이전이나 임대인의 재정 문제 등으로부터 안전을 확보할 수 있는 중요한 수단인데, 이 조항은 그런 보호를 받지 못하도록 막고 있습니다.
    따라서, 임차인 입장에서는 이 조항이 불합리할 수 있다는 점을 인지하고, 계약서를 주의 깊게 검토하고 협의해볼 필요가 있습니다.

    <질문>
    
    주어진 조항: "{clause}"

    용어 설명: {term_explanations}

    유사 과거 조항: "{corr_ex}"

    유사 과거 조항에 대한 해석: {judgment}

    답변:
    """
      explanation_prompt = PromptTemplate(template=explanation_template, input_variables=["clause", "term_explanations","corr_ex","judgment"])

    
    chain = LLMChain(prompt=explanation_prompt, llm=llm)

    if not detection:
        simplified_clause = chain.invoke({"clause": clause, "term_explanations": term_explanations})['text']
    else:
        simplified_clause = chain.invoke({"clause": clause, "term_explanations": term_explanations, "corr_ex": corr_ex, "judgment": judgment})['text']

    return simplified_clause


wikipedia.set_lang("ko")
tvly_key = "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"
tavily_client = TavilyClient(api_key=tvly_key)

def search_wikipedia(term):
    try:
        summary = wikipedia.summary(term, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return None
    except wikipedia.exceptions.PageError:
        return None

def search_tavily(term):
  response = tavily_client.search(f"법률 혹은 부동산 용어 {term}에 대해 설명해줘",max_results = 1)
  return response

def explain_legal_term(term):
  if (terms_df.term == term).sum() == 0:
    model = 'solar-1-mini-chat'
    llm = ChatUpstage(model=model, upstage_api_key=api_key)
    # Wikipedia에서 먼저 정보 검색
    wikipedia_info = search_wikipedia(term)

    # 위키피디아 정보가 있으면 LLM에게 쉽게 설명 요청
    if wikipedia_info:
        info = f"검색 정보: {wikipedia_info}"
        prompt = f"""다음 법률 혹은 부동산 용어에 대해 일반인이 쉽게 이해할 수 있도록 설명해.
        용어: {term}
        {info}"""
        return llm(prompt).content
    else:
        tvly_info = search_tavily(term)['results'][0]['content']
        info = f"검색 정보: {tvly_info}"
        prompt = f"""다음 법률 혹은 부동산 용어에 대해 일반인이 쉽게 이해할 수 있도록 설명해.
        용어: {term}
        {info}"""
        return llm(prompt).content
  else: return terms_df[terms_df.term == term]["definition"].values[0]