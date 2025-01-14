import tempfile
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, Document
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit 페이지 설정
st.set_page_config(
    page_title="벼락치기 도우미",
    page_icon="⏳",
)

# 파일에서 텍스트 추출
def extract_text_from_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
        tmp_file.write(file.read())
        temp_file_path = tmp_file.name

    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_file_path)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_file_path)
    elif file.name.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(temp_file_path)
    else:
        st.warning(f"지원하지 않는 파일 형식입니다: {file.name}")
        return ""

    documents = loader.load_and_split()
    full_text = " ".join([doc.page_content for doc in documents])

    os.remove(temp_file_path)
    return full_text

# 텍스트 청크로 분할
def split_text_into_chunks(uploaded_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # 청크 크기 조정
        chunk_overlap=300  # 청크 중첩 증가
    )
    documents = [Document(page_content=text) for text in uploaded_text.values()]
    return text_splitter.split_documents(documents)

# 벡터 저장소 생성
def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    return FAISS.from_documents(text_chunks, embeddings)

# 텍스트 요약 (문단별로 처리 및 개선)
def summarize_text(text_chunks, llm, max_summary_length=2000):
    def process_chunk(chunk):
        text = chunk.page_content
        messages = [
            SystemMessage(content="당신은 유능한 한국어 요약 도우미입니다. 요약은 간결하고 핵심 내용을 중심으로 작성해야 합니다."),
            HumanMessage(content=f"다음 텍스트를 한국어로 요약해주세요. 각 요약 항목은 핵심 내용을 포함하며 '-'로 시작하도록 작성해주세요:\n\n{text}")
        ]
        response = llm(messages)
        return response.content.strip()

    with ThreadPoolExecutor(max_workers=4) as executor:  # 병렬 처리
        summaries = list(executor.map(process_chunk, text_chunks))

    # 요약 항목을 빈 줄로 구분하여 결합
    combined_summary = "\n\n".join([f"- {summary}" for summary in summaries])
    return combined_summary[:max_summary_length] + "..." if len(combined_summary) > max_summary_length else combined_summary

# 공부 로드맵 생성
def create_study_roadmap(summary, llm, days_left, max_summary_length=2000):
    if len(summary) > max_summary_length:
        summary = summary[:max_summary_length] + "..."
    messages = [
        SystemMessage(content="당신은 한국 대학생을 위한 유능한 공부 로드맵 작성 도우미입니다."),
        HumanMessage(content=f"""
        다음 텍스트를 기반으로 {days_left}일 동안 한국 대학생들이 효과적으로 공부할 수 있는 계획을 작성해주세요:
        {summary}
        """)
    ]
    response = llm(messages)
    return response.content

# 예상 문제 생성
def generate_quiz_questions(summary, llm):
    messages = [
        SystemMessage(content="당신은 한국 대학생을 위한 예상 문제를 작성하는 도우미입니다."),
        HumanMessage(content=f"""
        다음 텍스트를 기반으로 중요도를 표시한 10개 이상의 예상 문제를 작성해주세요:
        {summary}
        - 예상 문제는 명확하고 구체적으로 작성해주세요.
        - 각 문제에는 중요도를 '높음', '중간', '낮음'으로 표시해주세요.
        """)
    ]
    response = llm(messages)
    return response.content

# Streamlit 앱 설정
def main():
    st.title("⏳ 대학생 벼락치기 도우미")

    if "uploaded_text" not in st.session_state:
        st.session_state.uploaded_text = {}

    if "roadmap" not in st.session_state:
        st.session_state.roadmap = None

    if "quiz" not in st.session_state:
        st.session_state.quiz = None

    if "summary" not in st.session_state:
        st.session_state.summary = {}  # 수정: 요약을 파일별로 저장하도록 초기화

    with st.sidebar:
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        problem_files = st.file_uploader("❓ 문제 업로드 (선택 사항)", type=["pdf", "docx", "pptx"], accept_multiple_files=True)  # 새로운 업로드란
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        exam_date = st.date_input("📅 시험 날짜를 선택하세요")
        process_button = st.button("🚀 벼락치기 시작하기")
        create_summary = st.checkbox("핵심 요약 생성", value=True)
        create_roadmap = st.checkbox("공부 로드맵 생성", value=True)
        create_quiz = st.checkbox("예상 문제 생성", value=True)
        generate_related_problems = st.checkbox("연관 문제 생성 (문제 업로드 시)", value=True)  # 새로운 체크박스 추가
        selected_files = st.multiselect("요약을 확인할 파일을 선택하세요:", list(st.session_state.summary.keys()))
        
        # 파일별 요약 결과 선택 (다중 선택 가능)

    if process_button:
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
            return
        if not uploaded_files:
            st.warning("강의 자료를 업로드해주세요!")
            return
        if not exam_date:
            st.warning("시험 날짜를 선택해주세요!")
            return

        # 시험까지 남은 기간 계산
        days_left = (exam_date - datetime.now().date()).days
        if days_left <= 0:
            st.warning("시험 날짜는 오늘보다 이후여야 합니다!")
            return

        # 업로드한 파일 텍스트 추출
        for file in uploaded_files:
            st.session_state.uploaded_text[file.name] = extract_text_from_file(file)

        # 벡터 저장소 및 요약 생성
        text_chunks = split_text_into_chunks(st.session_state.uploaded_text)
        vectorstore = create_vectorstore(text_chunks)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")  # GPT-4 유지

        # 선택적으로 단계 실행
        if create_summary:
            st.session_state.summary = {
                file_name: summarize_text([chunk], llm)
                for file_name, chunk in zip(st.session_state.uploaded_text.keys(), text_chunks)
            }
        if create_roadmap:
            st.session_state.roadmap = create_study_roadmap(
                "\n".join(st.session_state.summary.values()), llm, days_left
            )
        if create_quiz:
            st.session_state.quiz = generate_quiz_questions(
                "\n".join(st.session_state.summary.values()), llm
            )
        # 문제 텍스트를 최대 길이로 자르는 함수
        def truncate_text(text, max_tokens=2000):
            """텍스트를 최대 토큰 수로 제한합니다."""
            tokens = text.split()  # 단어 단위로 나눕니다
            if len(tokens) > max_tokens:
                return " ".join(tokens[:max_tokens])
            return text

    if st.session_state.uploaded_text:
        if create_summary and selected_files:
            st.subheader("📌 핵심 요약")
            for selected_file in selected_files:
                st.markdown(f"**파일명: {selected_file}**")
                st.markdown(st.session_state.summary[selected_file].replace("\n", "\n\n"))

        if create_roadmap:
            st.subheader("📋 공부 로드맵")
            st.markdown(st.session_state.roadmap)
        if create_quiz:
            st.subheader("❓ 예상 문제")
            st.markdown(st.session_state.quiz)

        if problem_files and generate_related_problems:
            st.subheader("📝 중간&기말 관련 문제 및 풀이")
            for file in problem_files:
                problem_text = extract_text_from_file(file)
                if problem_text:
                    # 문제 텍스트 자르기
                    truncated_problem_text = truncate_text(problem_text, max_tokens=2000)
                    messages = [
                        SystemMessage(content="당신은 한국 대학생을 위한 문제 생성 및 풀이 작성 도우미입니다."),
                        HumanMessage(content=f"""
                        다음 텍스트를 기반으로 최소 5개의 학문적 수준의 문제를 작성하고, 각 문제에 대한 보기와 풀이는 다음 형식으로 제공해주세요:
                        1. 문제: [문제 내용]
                        보기:
                        A) [보기1]
                        B) [보기2]
                        C) [보기3]
                        D) [보기4]
                        풀이: [문제 풀이]
                        텍스트:
                        {truncated_problem_text}
                        """)
                    ]
                    try:
                        # 모델 호출
                        response = llm(messages)
                        generated_questions = response.content.split("\n\n")  # 문제와 풀이 분리
                        st.markdown(f"**파일명: {file.name}**")

                        # 생성된 문제 출력
                        for idx, question_block in enumerate(generated_questions, 1):
                            # 문제와 풀이 분리
                            parts = question_block.split("풀이:")
                            question_text = parts[0].strip()
                            solution_text = parts[1].strip() if len(parts) > 1 else "풀이 없음"

                            # 문제 본문 출력
                            st.markdown(f"### 문제 {idx}")
                            st.markdown(question_text.replace("보기:", "\n\n**보기:**"))

                            # 풀이 숨기기
                            with st.expander(f"풀이 보기 (문제 {idx})"):
                                st.markdown(solution_text)
                    except Exception as e:
                        st.error(f"오류 발생: {e}")



        
if __name__ == "__main__":
    main()
