from utils import *
from dotenv import load_dotenv
import os

pdf_dir = "data/pdf"
files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir)]

text = get_pdf_text(files)

chunks = get_text_chunks(text)

# print("Text:", text)
for i, chunk in enumerate(chunks):
    print(f"\n\nChunk {i}", chunk)

print("Chunks: ", len(chunks))
