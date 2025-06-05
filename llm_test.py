from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from nltk.tokenize import sent_tokenize
import nltk
import fitz
nltk.download("punkt")

# === Load all PDFs ===
text_chunks = []
for pdf_path in Path("./test").glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])

    # === Chunk into sentences ===
    sentences = sent_tokenize(full_text)
    text_chunks.extend(sentences)

print(f"✅ Loaded {len(text_chunks)} sentences from PDFs.")

# === Set up local LLM ===
llm = Ollama(model="mistral")  # use a small model

# === Define food-related query ===
prompt_template = "Is this sentence about food in any way? Return 'yes' or 'no'. Sentence: \"{}\""

# === Run food detection ===
food_sentences = []
for sentence in text_chunks:
    response = llm.invoke(prompt_template.format(sentence))
    if "yes" in response.lower():
        food_sentences.append(sentence)

# === Print results ===
print(f"\n🍽️ Found {len(food_sentences)} food-related sentences:\n")
for s in food_sentences:
    print(f"• {s}")
