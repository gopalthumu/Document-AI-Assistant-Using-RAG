import os
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing")


DOCUMENT_PATH = Path("Document/Compendium.pdf")
INDEX_PATH = "faiss_index"


def load_document():
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {DOCUMENT_PATH}"
        )

    loader = PyPDFLoader(str(DOCUMENT_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")
    return documents



def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    return chunks



def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    vector_store.save_local(INDEX_PATH)

    print("Embeddings generated locally and stored in FAISS")
    return vector_store


def answer_question(vector_store, question):
    relevant_docs = vector_store.similarity_search(
        question,
        k=4,
    )

    if not relevant_docs:
        return "No relevant information was found."

    context = " ".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}"
        f"{doc.page_content}"
        for doc in relevant_docs
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a document question-answering assistant.
            Context:
            {context}"""
        ),
        ("human", "{question}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    print("Retrieved sources:")
    for doc in relevant_docs:
        print(
            f"- Page {doc.metadata.get('page', 'Unknown')}"
        )

    if isinstance(response.content, str):
        return response.content

    return " ".join(
        block["text"]
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def main():
    documents = load_document()
    chunks = split_document(documents)
    vector_store = create_vector_store(chunks)

    print("RAG application is ready!")
    print("Type 'exit' to stop.")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            answer = answer_question(
                vector_store,
                question,
            )
            print(" Answer: ", answer)

        except Exception as error:
            print(f"Error answering question: {error}")

if __name__ == "__main__":
    main()
