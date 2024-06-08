from utils import *


bo_path = "data/pdf/bo-2022.pdf"
text = extract_text_from_pdf(bo_path)

print("Text:", text)
