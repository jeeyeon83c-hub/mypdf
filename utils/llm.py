"""OpenAI 호출: 분석(단일/Map-Reduce), 문서 기반 Q&A, 재시도·오류 메시지 처리."""

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import streamlit as st
from openai import OpenAI

from utils import prompts
from utils.pdf_reader import chunk_label, doc_char_count, doc_full_text, make_chunks, select_chunks

# 이보다 짧은 입력은 한 번의 호출로 분석, 길면 Map-Reduce
MAX_SINGLE_CHARS = 100_000
# Q&A에 넣을 문서 발췌 최대 글자 수
QA_CONTEXT_CHARS = 30_000
# Q&A에 함께 보낼 이전 대화 턴 수
QA_HISTORY_TURNS = 6

# Map 단계 동시 호출 수
MAP_WORKERS = 4

MAX_RETRIES = 3
RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> OpenAI:
    # 재시도는 _create_with_retry에서 직접 처리한다.
    return OpenAI(api_key=api_key, timeout=180, max_retries=0)


def _is_quota_error(e: Exception) -> bool:
    """크레딧 소진(insufficient_quota)은 기다려도 풀리지 않으므로 재시도하지 않는다."""
    return isinstance(e, openai.RateLimitError) and getattr(e, "code", None) == "insufficient_quota"


def _create_with_retry(client: OpenAI, **kwargs):
    """지수 백오프(1s, 2s, 4s)로 최대 MAX_RETRIES회 재시도."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RETRYABLE as e:
            if attempt == MAX_RETRIES or _is_quota_error(e):
                raise
            time.sleep(2**attempt)


def _complete(client: OpenAI, model: str, messages: list[dict]) -> str:
    resp = _create_with_retry(client, model=model, messages=messages)
    return resp.choices[0].message.content or ""


def _stream(client: OpenAI, model: str, messages: list[dict]) -> Iterator[str]:
    stream = _create_with_retry(client, model=model, messages=messages, stream=True)
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def build_task(analysis_type: str, custom_prompt: str = "", combined: bool = False) -> str:
    task = custom_prompt.strip() if analysis_type == prompts.CUSTOM else prompts.ANALYSIS_TYPES[analysis_type]
    if combined:
        task += prompts.COMBINED_SUFFIX
    return task


def analyze(
    client: OpenAI,
    model: str,
    docs: list[dict],
    task: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> Iterator[str]:
    """문서(들)를 분석해 결과를 스트리밍으로 돌려준다.

    전체 텍스트가 MAX_SINGLE_CHARS 이하면 한 번에, 넘으면 청크별 메모(Map) → 종합(Reduce).
    on_progress(done, total)은 Map 단계 진행률 콜백.
    """
    total_chars = sum(doc_char_count(d) for d in docs)

    if total_chars <= MAX_SINGLE_CHARS:
        text = "\n\n".join(f"=== 문서: {d['name']} ===\n{doc_full_text(d)}" for d in docs)
        messages = [_system(prompts.SYSTEM_BASE), _user(prompts.SINGLE_PROMPT.format(task=task, text=text))]
        yield from _stream(client, model, messages)
        return

    chunks = [c for d in docs for c in make_chunks(d)]

    def map_one(c: dict) -> str:
        label = chunk_label(c)
        content = prompts.MAP_PROMPT.format(label=label, task=task, text=c["text"])
        note = _complete(client, model, [_system(prompts.SYSTEM_BASE), _user(content)])
        return f"### [{label}]\n{note}"

    # Map: 청크별 메모를 병렬로 만든다. 진행률 콜백은 Streamlit 요소를 건드리므로 메인 스레드에서만 호출.
    notes: list[str | None] = [None] * len(chunks)
    if on_progress:
        on_progress(0, len(chunks))
    with ThreadPoolExecutor(max_workers=MAP_WORKERS) as pool:
        futures = {pool.submit(map_one, c): i for i, c in enumerate(chunks)}
        for done, fut in enumerate(as_completed(futures), start=1):
            notes[futures[fut]] = fut.result()  # 실패 시 예외가 그대로 올라간다
            if on_progress:
                on_progress(done, len(chunks))

    # 메모가 너무 길면 묶음별로 중간 종합을 반복해 최종 입력을 MAX_SINGLE_CHARS 이하로 줄인다.
    notes = _shrink_notes(client, model, task, notes)

    content = prompts.REDUCE_PROMPT.format(task=task, notes="\n\n".join(notes))
    yield from _stream(client, model, [_system(prompts.SYSTEM_BASE), _user(content)])


def _shrink_notes(client: OpenAI, model: str, task: str, notes: list[str]) -> list[str]:
    for _ in range(5):  # 안전장치: 무한 반복 방지
        if sum(len(n) for n in notes) <= MAX_SINGLE_CHARS or len(notes) <= 1:
            return notes
        batches, cur, cur_len = [], [], 0
        for n in notes:
            if cur and cur_len + len(n) > MAX_SINGLE_CHARS:
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append(n)
            cur_len += len(n)
        batches.append(cur)
        if len(batches) == len(notes):  # 메모 하나하나가 이미 한도 근처 → 더 줄일 수 없음
            return notes
        notes = [
            _complete(client, model, [
                _system(prompts.SYSTEM_BASE),
                _user(prompts.REDUCE_PROMPT.format(task=task, notes="\n\n".join(b))
                      + "\n\n(이 결과는 다시 종합될 중간 메모입니다. 페이지 번호를 유지하세요.)"),
            ])
            for b in batches
        ]
    return notes


def answer_question(
    client: OpenAI,
    model: str,
    docs: list[dict],
    question: str,
    history: list[dict],
) -> tuple[Iterator[str], list[str]]:
    """질문과 관련된 청크를 골라 답변을 스트리밍한다. (답변 스트림, 사용한 발췌 라벨) 반환."""
    # Q&A는 키워드 매칭 정확도를 위해 작은 청크로 나눈다.
    chunks = [c for d in docs for c in make_chunks(d, max_chars=4_000)]
    picked = select_chunks(chunks, question, QA_CONTEXT_CHARS)
    context = "\n\n".join(f"--- [{chunk_label(c)}] ---\n{c['text']}" for c in picked)

    messages = [_system(prompts.QA_SYSTEM)]
    for m in history[-QA_HISTORY_TURNS * 2 :]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append(_user(prompts.QA_USER.format(context=context, question=question)))

    return _stream(client, model, messages), [chunk_label(c) for c in picked]


def friendly_error(e: Exception) -> str:
    """OpenAI 예외를 사용자용 한국어 메시지로 바꾼다."""
    if isinstance(e, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. `.env` 또는 Streamlit Secrets의 OPENAI_API_KEY를 확인하세요."
    if isinstance(e, openai.PermissionDeniedError):
        return "이 API 키로는 해당 모델을 사용할 권한이 없습니다."
    if isinstance(e, openai.NotFoundError):
        return "모델을 찾을 수 없습니다. OPENAI_MODEL 설정값(모델 이름)을 확인하세요."
    if _is_quota_error(e):
        return "OpenAI 크레딧(사용 한도)이 소진되었습니다. OpenAI 계정의 결제·사용량 설정을 확인하세요."
    if isinstance(e, openai.RateLimitError):
        return "요청 한도를 초과했습니다(Rate limit 또는 크레딧 부족). 잠시 후 다시 시도하세요."
    if isinstance(e, openai.APITimeoutError):
        return "응답 시간이 초과되었습니다. 문서가 너무 길 수 있으니 잠시 후 다시 시도하세요."
    if isinstance(e, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하세요."
    if isinstance(e, openai.BadRequestError):
        return f"요청이 거부되었습니다. 문서가 너무 길거나 요청 형식에 문제가 있습니다. ({e.message})"
    if isinstance(e, openai.APIStatusError):
        return f"OpenAI 서버 오류가 발생했습니다 (HTTP {e.status_code}). 잠시 후 다시 시도하세요."
    return f"알 수 없는 오류가 발생했습니다: {e}"
