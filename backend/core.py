import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone

INDEX_NAME = "langchain-doc-index"
_PINECONE_INITIALIZED = False


def _init_pinecone() -> None:
    global _PINECONE_INITIALIZED
    if _PINECONE_INITIALIZED:
        return

    api_key = os.environ.get("PINECONE_API_KEY")
    environment = os.environ.get("PINECONE_ENVIRONMENT_REGION")
    missing = [
        name
        for name, value in [
            ("PINECONE_API_KEY", api_key),
            ("PINECONE_ENVIRONMENT_REGION", environment),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    pinecone.init(api_key=api_key, environment=environment)
    _PINECONE_INITIALIZED = True


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")

    _init_pinecone()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
        openai_api_key=openai_api_key,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})
