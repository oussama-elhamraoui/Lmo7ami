from utils import *
import os
from dotenv import load_dotenv

load_dotenv()

pdf_dir = "data/pdf"
files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir)]

text = get_pdf_text(files)

for file in os.listdir(pdf_dir):
    text = get_pdf_file(os.path.join(pdf_dir, file))
    chunks = get_text_chunks(text)
    # vectorstore = FAISS.load_local("model", embeddings=embeddings, allow_dangerous_deserialization=True)
    embeddings = OpenAIEmbeddings(chunk_size=1024, show_progress_bar=True)
    try: 
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        vectorstore.save_local("model", index_name=file)
    except:
        continue

# print(vectorstore)

# chat = get_conversation_chain(vectorstore)

# response = chat({"question": "Quels sont les créances publiques qui sont perçues"})

# print(response)
