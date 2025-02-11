from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langchain_milvus.vectorstores import Milvus
import pandas as pd



def load_passages() -> Milvus:
    """
    Loads passages, chunks them, loads in vector db, returns ensemble retriever
    """
    passages = pd.read_csv("./dataset/passages.csv")
    all_documents = []
    for i in range(len(passages["passage"])): 
        doc = Document(page_content=str(passages["passage"][i]))
        all_documents.append(doc)

    print("Number of passages to load in the database: ", len(passages["passage"]))

    embedding_model = HuggingFaceEmbeddings(model_name="embeddings/gte-base-en-v1.5", model_kwargs={"device": "cuda", "trust_remote_code": True})
    
    vectordb = Milvus.from_documents(
            all_documents,
            embedding_model,
            connection_args={"uri": "./db/milvus.db"},
            collection_name="langchain_example",
            index_params={"metric_type": "COSINE"},
            auto_id = False
        )
    print("Documents loaded in the vector db!")
    return vectordb


def load_qa()->pd.DataFrame:
    """
    Loads the question and answer dataset
    """
    qa_data = pd.read_csv("./dataset/dataset_qa.csv")
    print(qa_data.columns)
    return qa_data





