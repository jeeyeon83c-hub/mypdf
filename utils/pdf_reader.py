"""PDF 텍스트 추출과 청크 분할. 업로드 파일은 디스크에 저장하지 않고 메모리에서만 처리한다."""

import math
import re
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

# 페이지당 평균 글자 수가 이보다 적으면 스캔본(이미지 PDF)으로 본다.
MIN_CHARS_PER_PAGE = 20

# 청크 하나의 최대 글자 수 (페이지 단위로 묶되 이 크기를 넘지 않게)
CHUNK_CHARS = 30_000

# 추출 결과 캐시 크기 (Streamlit Cloud 메모리 한도를 고려해 상한을 둔다)
CACHE_MAX_ENTRIES = 32

_DEPENDENCY_MSG = "AES로 암호화된 PDF를 읽으려면 cryptography 패키지가 필요합니다 (requirements.txt 확인)."


@st.cache_data(show_spinner=False, max_entries=CACHE_MAX_ENTRIES)
def extract_pdf(file_bytes: bytes, file_name: str) -> dict:
    """PDF bytes에서 페이지별 텍스트를 추출한다.

    반환: {name, size, num_pages, pages: [str], error: str|None, warning: str|None}
    """
    result = {
        "name": file_name,
        "size": len(file_bytes),
        "num_pages": 0,
        "pages": [],
        "error": None,
        "warning": None,
    }

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except PdfReadError as e:
        result["error"] = f"PDF를 읽을 수 없습니다: {e}"
        return result
    except Exception as e:  # 손상된 파일 등
        result["error"] = f"PDF를 여는 중 오류가 발생했습니다: {e}"
        return result

    if reader.is_encrypted:
        # 빈 암호로 열리는 PDF(열람 제한 없음)는 계속 처리한다.
        try:
            if not reader.decrypt(""):
                result["error"] = "암호가 걸린 PDF는 지원하지 않습니다."
                return result
        except DependencyError:
            result["error"] = _DEPENDENCY_MSG
            return result
        except Exception:
            result["error"] = "암호가 걸린 PDF는 지원하지 않습니다."
            return result

    pages = []
    for page in reader.pages:
        try:
            # 일부 PDF는 공백 자리에 NUL 문자를 넣어 두므로 공백으로 바꾼다.
            pages.append((page.extract_text() or "").replace("\x00", " ").strip())
        except DependencyError:
            # 편집만 제한된 AES PDF는 decrypt("")는 통과하고 페이지를 읽을 때 실패한다.
            result["error"] = _DEPENDENCY_MSG
            return result
        except Exception:
            pages.append("")

    result["pages"] = pages
    result["num_pages"] = len(pages)

    total_chars = sum(len(p) for p in pages)
    if not pages:
        result["error"] = "페이지가 없는 PDF입니다."
    elif total_chars == 0:
        result["error"] = "텍스트를 추출할 수 없습니다(스캔 PDF 가능성). 분석 대상에서 제외합니다."
    elif total_chars < MIN_CHARS_PER_PAGE * len(pages):
        result["warning"] = (
            "추출된 텍스트가 거의 없습니다. 스캔(이미지) PDF일 수 있으며, "
            "이 경우 분석이 정확하지 않을 수 있습니다."
        )

    return result


def doc_char_count(doc: dict) -> int:
    return sum(len(p) for p in doc["pages"])


def make_chunks(doc: dict, max_chars: int = CHUNK_CHARS) -> list[dict]:
    """문서를 페이지 단위로 묶어 청크로 나눈다.

    반환: [{name, start_page, end_page, text}] — text에는 [p.N] 페이지 표시가 들어 있다.
    한 페이지가 max_chars보다 길면 그 페이지를 여러 청크로 쪼갠다.
    """
    chunks: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    start = None

    def flush(end_page: int):
        nonlocal buf, buf_len, start
        if buf:
            chunks.append(
                {"name": doc["name"], "start_page": start, "end_page": end_page, "text": "\n\n".join(buf)}
            )
        buf, buf_len, start = [], 0, None

    for page_no, text in enumerate(doc["pages"], start=1):
        if not text:
            continue
        block = f"[p.{page_no}]\n{text}"

        if len(block) > max_chars:
            flush(page_no - 1)
            for i in range(0, len(text), max_chars):
                chunks.append(
                    {
                        "name": doc["name"],
                        "start_page": page_no,
                        "end_page": page_no,
                        "text": f"[p.{page_no}]\n{text[i:i + max_chars]}",
                    }
                )
            continue

        if buf and buf_len + len(block) > max_chars:
            flush(page_no - 1)
        if start is None:
            start = page_no
        buf.append(block)
        buf_len += len(block) + 2
        last_page = page_no

    if buf:
        flush(last_page)
    return chunks


def chunk_label(chunk: dict) -> str:
    s, e = chunk["start_page"], chunk["end_page"]
    pages = f"p.{s}" if s == e else f"p.{s}-{e}"
    return f"{chunk['name']}, {pages}"


def doc_full_text(doc: dict) -> str:
    """[p.N] 표시를 붙인 문서 전체 텍스트."""
    return "\n\n".join(f"[p.{i}]\n{t}" for i, t in enumerate(doc["pages"], start=1) if t)


# --- Q&A용 키워드 기반 청크 선택 ---

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _terms(text: str) -> set[str]:
    """질문에서 검색어를 뽑는다. 한국어는 조사가 붙으므로 2글자 n-gram도 함께 쓴다."""
    terms = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) >= 2:
            terms.add(tok)
        if re.search(r"[가-힣]", tok) and len(tok) > 2:
            terms.update(tok[i:i + 2] for i in range(len(tok) - 1))
    return terms


# 최고 점수 대비 이 비율 미만인 청크는 관련 없다고 보고 제외
MIN_RELATIVE_SCORE = 0.25


def select_chunks(chunks: list[dict], question: str, max_chars: int) -> list[dict]:
    """질문 키워드와 관련도가 높은 청크를 max_chars 안에서 고른다.

    점수 = Σ (등장 횟수, 최대 5회) × 검색어 길이 × IDF.
    IDF 덕분에 거의 모든 청크에 나오는 흔한 2글자 조각('구간', '센터' 등)은 점수에 거의 기여하지 않는다.
    """
    terms = _terms(question)
    lows = [c["text"].lower() for c in chunks]
    n = len(chunks)
    idf = {}
    for t in terms:
        df = sum(1 for low in lows if t in low)
        if df:
            idf[t] = math.log((n + 1) / df)

    if not idf:
        scored = [(0.0, i) for i in range(n)]
    else:
        scored = []
        for i, low in enumerate(lows):
            score = sum(min(low.count(t), 5) * len(t) * w for t, w in idf.items())
            scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[0][0]
        matched = [s for s in scored if top > 0 and s[0] >= top * MIN_RELATIVE_SCORE]
        # 전부 0점(모든 청크에 똑같이 나오는 단어뿐)이면 앞부분 청크로 대체
        scored = matched or sorted(scored, key=lambda x: x[1])

    picked, total = [], 0
    for _, i in scored:
        c = chunks[i]
        if total + len(c["text"]) > max_chars and picked:
            continue
        picked.append(c)
        total += len(c["text"])
        if total >= max_chars:
            break
    # 원래 문서 순서대로 정렬해 문맥이 자연스럽게
    order = {id(c): i for i, c in enumerate(chunks)}
    return sorted(picked, key=lambda c: order[id(c)])


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
