from datetime import datetime

import streamlit as st

from utils import prompts
from utils.config import get_api_key, get_model
from utils.llm import analyze, answer_question, build_task, friendly_error, get_client
from utils.pdf_reader import doc_char_count, extract_pdf, format_size

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

TAB_ANALYSIS, TAB_QA, TAB_RAW = "📊 분석", "💬 Q&A", "📃 원문"

st.title("📄 PDF 분석기")
st.caption("PDF를 올리면 AI가 요약·핵심 정리·수치 추출을 하고, 문서 내용에 대한 질문에 답합니다.")

# --- 설정 확인 ---
api_key = get_api_key()
if not api_key:
    st.error(
        "OpenAI API 키가 설정되지 않았습니다.\n\n"
        "- 로컬: 프로젝트 폴더의 `.env`에 `OPENAI_API_KEY`를 입력하세요.\n"
        "- Streamlit Cloud: 앱 설정 > Secrets에 `OPENAI_API_KEY`를 입력하세요."
    )
    st.stop()

model = get_model()
client = get_client(api_key)

st.session_state.setdefault("files_key", None)
st.session_state.setdefault("results", None)  # {"meta": {...}, "sections": [{"title", "content"}]}
st.session_state.setdefault("messages", [])  # [{"role", "content", "sources"}]

# --- 사이드바 1: 업로드 ---
with st.sidebar:
    st.header("📁 PDF 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일 (여러 개 가능)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

def md_escape(text: str) -> str:
    """파일명 등 사용자 입력을 Markdown/HTML로 해석되지 않게 이스케이프한다."""
    out = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for ch in "\\`*_{}[]()#+-.!|~$":
        out = out.replace(ch, "\\" + ch)
    return out


# 업로드 파일이 바뀌면 이전 결과와 대화를 초기화
# (file_id는 업로드할 때마다 새로 발급되므로 이름·크기가 같은 파일로 바꿔도 감지된다)
files_key = tuple(f.file_id for f in uploaded_files or [])
if st.session_state.files_key != files_key:
    st.session_state.files_key = files_key
    st.session_state.results = None
    st.session_state.messages = []

docs = []
if uploaded_files:
    with st.spinner("PDF에서 텍스트를 추출하는 중..."):
        seen: dict[str, int] = {}
        for f in uploaded_files:
            d = extract_pdf(f.getvalue(), f.name)
            # 이름이 같은 파일이 여러 개면 "이름 (2)"처럼 구분한다.
            seen[f.name] = seen.get(f.name, 0) + 1
            if seen[f.name] > 1:
                d = {**d, "name": f"{f.name} ({seen[f.name]})"}
            docs.append(d)
valid_docs = [d for d in docs if not d["error"]]

# --- 사이드바 2: 파일 목록 · 분석 설정 ---
with st.sidebar:
    if docs:
        for d in docs:
            icon = "❌" if d["error"] else ("⚠️" if d["warning"] else "✅")
            st.markdown(f"{icon} **{md_escape(d['name'])}**")
            st.caption(f"{d['num_pages']}쪽 · {format_size(d['size'])}")

    st.divider()
    st.header("⚙️ 분석 설정")
    analysis_type = st.selectbox("분석 유형", list(prompts.ANALYSIS_TYPES))
    custom_prompt = ""
    if analysis_type == prompts.CUSTOM:
        custom_prompt = st.text_area("분석 요청", placeholder="예: 이 문서의 위험 요소와 대응 방안을 정리해줘")

    mode = "개별 분석"
    if len(valid_docs) > 1:
        mode = st.radio(
            "여러 문서 분석 방식",
            ["개별 분석", "통합 분석"],
            help="개별 분석: 문서마다 따로 분석 · 통합 분석: 모든 문서를 함께 분석하고 비교",
            horizontal=True,
        )

    run = st.button("분석 시작", type="primary", icon="🚀", width="stretch", disabled=not valid_docs)

    st.divider()
    st.caption(f"🤖 모델: `{model}`")

# --- 업로드 전: 사용 안내 ---
if not uploaded_files:
    st.write("")
    cols = st.columns(3)
    guide = [
        ("1️⃣ PDF 업로드", "왼쪽 사이드바에 PDF를 끌어다 놓으세요. 여러 개를 한 번에 올릴 수 있습니다."),
        ("2️⃣ 분석 유형 선택", "요약, 핵심 포인트, 키워드, 수치 추출 중 고르거나 직접 요청을 입력하세요."),
        ("3️⃣ 결과 확인 · 질문", "분석 결과를 내려받고, Q&A 탭에서 문서 내용을 자유롭게 질문하세요."),
    ]
    for col, (title, body) in zip(cols, guide):
        with col.container(border=True):
            st.markdown(f"#### {title}")
            st.write(body)
    st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드해 시작하세요.")
    st.stop()

# --- 파일 요약 지표 ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("업로드 파일", f"{len(docs)}개")
c2.metric("분석 가능", f"{len(valid_docs)}개")
c3.metric("총 페이지", f"{sum(d['num_pages'] for d in valid_docs):,}쪽")
c4.metric("추출 글자 수", f"{sum(doc_char_count(d) for d in valid_docs):,}자")

for d in docs:
    if d["error"]:
        st.error(f"**{md_escape(d['name'])}** — {d['error']}", icon="❌")
    elif d["warning"]:
        st.warning(f"**{md_escape(d['name'])}** — {d['warning']}", icon="⚠️")

if not valid_docs:
    st.warning("분석할 수 있는 PDF가 없습니다. 텍스트가 포함된 PDF를 업로드하세요.")
    st.stop()

# 분석 시작을 누르면 분석 탭으로 이동
if run:
    st.session_state.main_tab = TAB_ANALYSIS
tab_analysis, tab_qa, tab_raw = st.tabs([TAB_ANALYSIS, TAB_QA, TAB_RAW], key="main_tab")


# --- 탭1: 분석 ---
def run_analysis():
    if analysis_type == prompts.CUSTOM and not custom_prompt.strip():
        st.warning("'사용자 지정'을 선택했다면 사이드바에 분석 요청 내용을 입력하세요.")
        return

    st.session_state.results = None
    combined = mode == "통합 분석" and len(valid_docs) > 1
    targets = [("통합 분석", valid_docs)] if combined else [(d["name"], [d]) for d in valid_docs]
    task = build_task(analysis_type, custom_prompt, combined=combined)

    sections, failed = [], 0
    for title, target_docs in targets:
        with st.container(border=True):
            st.markdown(f"### {md_escape(title)}")
            status = st.status(f"{analysis_type} 분석 중...", expanded=False)
            bar = status.empty()

            def on_progress(done, total, bar=bar, status=status):
                status.update(label=f"긴 문서를 나누어 읽는 중... ({done}/{total})", expanded=True)
                bar.progress(done / total)

            try:
                content = st.write_stream(analyze(client, model, target_docs, task, on_progress))
            except Exception as e:
                status.update(label="분석 실패", state="error", expanded=False)
                st.error(friendly_error(e))
                failed += 1
                continue
            bar.empty()
            status.update(label="분석 완료", state="complete", expanded=False)
            sections.append({"title": title, "content": content})

    if sections:
        st.session_state.results = {
            "meta": {
                "analysis_type": analysis_type,
                "custom_prompt": custom_prompt.strip(),
                "mode": "통합 분석" if combined else "개별 분석",
                "model": model,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "sections": sections,
        }
        st.toast("분석이 완료되었습니다." if not failed else f"일부 문서 분석에 실패했습니다 ({failed}건).",
                 icon="✅" if not failed else "⚠️")


def results_markdown(results: dict) -> str:
    meta = results["meta"]
    lines = [
        "# PDF 분석 결과",
        "",
        f"- 분석 유형: {meta['analysis_type']}",
        f"- 분석 방식: {meta['mode']}",
        f"- 모델: {meta['model']}",
        f"- 생성 시각: {meta['created']}",
    ]
    if meta["custom_prompt"]:
        lines.append(f"- 분석 요청: {meta['custom_prompt']}")
    for s in results["sections"]:
        lines += ["", f"## {s['title']}", "", s["content"]]
    return "\n".join(lines) + "\n"


with tab_analysis:
    if run:
        run_analysis()
    elif st.session_state.results:
        meta = st.session_state.results["meta"]
        st.caption(f"{meta['analysis_type']} · {meta['mode']} · {meta['created']}")
        for s in st.session_state.results["sections"]:
            with st.container(border=True):
                st.markdown(f"### {md_escape(s['title'])}")
                st.markdown(s["content"])
    else:
        st.info("사이드바에서 분석 유형을 고르고 **🚀 분석 시작**을 누르세요.")

    if st.session_state.results:
        st.download_button(
            "결과 다운로드 (.md)",
            data=results_markdown(st.session_state.results),
            file_name=f"pdf_analysis_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
            icon="⬇️",
        )

# --- 탭2: Q&A ---
def sources_caption(sources: list[str]) -> str:
    """'파일명, p.N' 라벨들을 파일별로 묶어 짧게 표시한다."""
    by_file: dict[str, list[str]] = {}
    for label in sources:
        name, _, pages = label.rpartition(", ")
        by_file.setdefault(name, []).append(pages.removeprefix("p."))
    parts = [f"{md_escape(name)} (p.{', '.join(pages)})" for name, pages in by_file.items()]
    return "📎 참고한 발췌: " + " · ".join(parts)


with tab_qa:
    chat_box = st.container()
    with chat_box:
        intro = st.empty()
        if not st.session_state.messages:
            intro.info("업로드한 문서 내용에 대해 질문하세요. 답변에 근거 페이지가 함께 표시됩니다.")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    st.caption(sources_caption(m["sources"]))

    question = st.chat_input("문서에 대해 질문하세요")
    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        intro.empty()
        with chat_box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("문서에서 답을 찾는 중..."):
                        stream, sources = answer_question(client, model, valid_docs, question, history)
                        answer = st.write_stream(stream)
                    st.caption(sources_caption(sources))
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    st.error(friendly_error(e))
                    st.session_state.messages.pop()  # 실패한 질문은 기록에서 제거

    if st.session_state.messages:
        if st.button("대화 지우기", icon="🗑️"):
            st.session_state.messages = []
            st.rerun()

# --- 탭3: 원문 ---
with tab_raw:
    idx = 0
    if len(valid_docs) > 1:
        idx = st.selectbox("파일 선택", range(len(valid_docs)), format_func=lambda i: valid_docs[i]["name"])
    doc = valid_docs[idx]
    st.caption(f"{doc['num_pages']}쪽 · {doc_char_count(doc):,}자")
    for i, text in enumerate(doc["pages"], start=1):
        with st.expander(f"{i}쪽", expanded=(i == 1)):
            st.text(text if text else "(추출된 텍스트 없음)")
