"""설정 로딩: Streamlit Secrets(배포) → .env / 환경변수(로컬) 순서로 읽는다."""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-6.1-terra"


def get_config(key: str, default: str | None = None) -> str | None:
    # st.secrets는 secrets.toml이 없으면 접근 시 예외를 던지므로 감싼다.
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    value = os.getenv(key)
    return value if value else default


def get_api_key() -> str | None:
    return get_config("OPENAI_API_KEY")


def get_model() -> str:
    return get_config("OPENAI_MODEL", DEFAULT_MODEL)
