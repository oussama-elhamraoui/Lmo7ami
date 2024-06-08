from utils import *
from dotenv import load_dotenv
import os
from langchain.chains import SimpleSequentialChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

pdf_dir = "data/pdf"
files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir)]

llm = OpenAI()

text = get_pdf_text(files)
prompt_template = PromptTemplate(
    input_variables=["text", "user_question"],
    template="Here is some text from a PDF: {text}\nThe user will ask you about it. User question: {user_question}",
)
chain = SimpleSequentialChain(prompt_template=prompt_template, llm=llm)

def get_response(pdf_text, user_question):
    response = chain.run({"text": pdf_text, "user_question": user_question})
    return response


# Example usage
user_question = "What is the main topic discussed in this document?"
response = get_response(text, user_question)
