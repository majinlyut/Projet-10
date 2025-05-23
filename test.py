from langchain import __version__ as langchain_version
import faiss
import mistral

print(f"LangChain version: {langchain_version}")
print(f"FAISS version: {faiss.__version__}")
print("Mistral module loaded successfully")
