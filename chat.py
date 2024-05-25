from openai import OpenAI
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader

client = OpenAI()

loader = CSVLoader(file_path="knowledge_source.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

retrieve_info("Olá, me explique o que é atendimento N1")


try:
    loader = CSVLoader(file_path="knowledge_source.csv")
    documents = loader.load()
    print("Documentos carregados com sucesso.")
except Exception as e:
    print("Erro ao carregar o arquivo: {e}")

print(documents)