import os

# Paste your brand-new key exactly as shown in Groq console (no spaces/newlines)
os.environ["GROQ_API_KEY"] = "*************************************************"

# Quick sanity check
print("Key set?", "GROQ_API_KEY" in os.environ)
print("Length:", len(os.environ["GROQ_API_KEY"]))
print("Starts with gsk_?", os.environ["GROQ_API_KEY"].startswith("gsk_"))




from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
print(llm.invoke("ping"))



# --- Install dependencies (Colab-safe) ---
%pip -q install -U langchain-core langchain-community langchain-text-splitters langchain-groq chromadb sentence-transformers pypdf

# --- Configuration (edit only the key if you want to skip prompt) ---
PDF_PATH = "/content/mental_health_Document.pdf"   # Your PDF path
PERSIST_DIR = "/content/chromadb_mental_health"    # Chroma persistence dir
EMBED_MODEL = "BAAI/bge-small-en-v1.5"             # Compact, strong English embeddings
GROQ_MODEL = "llama-3.3-70b-versatile"             # Groq model

# OPTIONAL: set your key here to skip the prompt (keep quotes)
# import os; os.environ["GROQ_API_KEY"] = "gsk_...Tgii"

# --- Imports ---
import os
from getpass import getpass

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# --- Ensure persistence directory exists ---
os.makedirs(PERSIST_DIR, exist_ok=True)

# --- GROQ API KEY (prompt if not set) ---
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter GROQ_API_KEY: ")

# --- Initialize LLM ---
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name=GROQ_MODEL,
    )

# --- Build vector database from a single PDF ---
def create_vector_db_from_pdf(pdf_path):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    if not os.path.isfile(pdf_path):
        print(f"PDF not found at {pdf_path}. Please upload it and re-run.")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    texts = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    return vectordb

def load_vector_db():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# --- RAG pipeline (no langchain.chains import) ---
def build_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a compassionate mental health chatbot. "
         "Use the provided context to answer accurately and be supportive. "
         "If the answer is not in the context, say you don't know and suggest general next steps; "
         "this is not medical advice."),
        ("human", "Context:\n{context}\n\nUser question: {question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- CLI loop ---
def run_cli(chain):
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("Human: ").strip()
        if q.lower() == "exit":
            print("Chatbot: Take care of yourself, Goodbye!")
            break
        if not q:
            print("Chatbot: Please ask a valid question.")
            continue
        try:
            ans = chain.invoke(q)
        except Exception as e:
            ans = f"Error: {e}"
        print(f"Chatbot: {ans}")

# --- Main ---
def main():
    print("Initializing Chatbot.........")
    llm = initialize_llm()
    has_index = os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR))
    vectordb = load_vector_db() if has_index else create_vector_db_from_pdf(PDF_PATH)
    rag_chain = build_rag_chain(vectordb, llm)
    run_cli(rag_chain)

if __name__ == "__main__":
    main()
