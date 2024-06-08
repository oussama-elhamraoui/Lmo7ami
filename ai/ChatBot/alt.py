# Import os to set API key
import os

# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Bring in streamlit for UI/app interface

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader

# Import chroma as the vector store
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from dotenv import load_dotenv

load_dotenv()


# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader("data/pdf/code de recouvrement des créances publiques.pdf")
# Split pages from pdf
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name="annualreport")

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store,
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)
