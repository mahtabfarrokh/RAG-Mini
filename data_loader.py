from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain.retrievers import EnsembleRetriever

import pandas as pd



def load_passages()->tuple[EnsembleRetriever, list[Document]]:
    """
    Loads passages, chunks them, loads in vector db, returns ensemble retriever
    """
    passages = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
    passages.reset_index(inplace=True)

    print(passages.head())

    all_documents = []
    for i in range(len(passages["passage"])): 
        doc = Document(page_content=passages["passage"][i], metadata={"id": passages["id"][i]})
        all_documents.append(doc)

    print(len(all_documents))

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(all_documents)

    bm25_retriever = BM25Retriever.from_documents(chunked_documents)
    bm25_retriever.k = 5

    embedding_model = HuggingFaceEmbeddings(model_name="embeddings/gte-large-en-v1.5", model_kwargs={"device": "remote", "trust_remote_code": True})
    vector_db = Chroma.from_documents(
        documents= chunked_documents, 
        embedding= embedding_model,
        persist_directory="./db/"
    )
    similarity_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, similarity_retriever], weights=[0.5, 0.5])

    return ensemble_retriever, chunked_documents



def load_qa()->pd.DataFrame:
    qa_data = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/test.parquet/part.0.parquet")
    qa_data.reset_index(inplace=True)
    return qa_data