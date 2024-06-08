from utils import *
from dotenv import load_dotenv
import os

load_dotenv()

pdf_dir = "data/pdf"
files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir)]
files = ["data/pdf/code de recouvrement des créances publiques.pdf"]

text = get_pdf_text(files)

chunks = get_text_chunks(text)


print("Chunks: ", len(chunks))

vectorstore = get_vectorstore(chunks)
