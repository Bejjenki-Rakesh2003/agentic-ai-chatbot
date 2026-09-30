# src/llm.py

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add it to your .env file locally or Streamlit Cloud Secrets."
    )


# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to Groq and return the generated response.
    """
    try:
        response = llm.invoke(prompt)

        if hasattr(response, "content"):
            return response.content

        return str(response)

    except Exception as e:
        raise RuntimeError(
            f"Groq LLM request failed: {str(e)}"
        ) from e