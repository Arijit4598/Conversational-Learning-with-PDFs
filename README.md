## Create Virtual Environment:#####
python -m venv myenv
myenv/scripts/activate

## RAG PIPELINE USING HUGGING-FACE TRANSFORMERS:
This is a Document Q&A / RAG (Retrieval-Augmented Generation) pipeline where:
📄 PDF → 🔎 Split & Embed → 📂 Store in FAISS → 🤖 Query with ChatGPT → 📊 Display in Streamlit.
