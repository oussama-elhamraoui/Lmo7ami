from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

def get_pdf_text(pdf_docs: list[str]):
    text = ""
    for pdf in pdf_docs:
        text += get_pdf_file(pdf)
    return text
def get_pdf_file(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text: str) -> list[str]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(text_chunks: list[str]) -> FAISS:
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    embeddings = OpenAIEmbeddings(chunk_size=1024, show_progress_bar=True, model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("model")
    return vectorstore


def get_conversation_chain(vectorstore: FAISS):
    llm = ChatOpenAI(temperature=0.9)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True).chat_memory.add_user_message
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain
