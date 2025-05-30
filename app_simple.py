import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load CSV and prepare documents
df = pd.read_csv("controls.csv")
controls = df.to_dict(orient="records")
texts = [f"{c['id']} ({c['framework']}): {c['description']}" for c in controls]
docs = [Document(page_content=text) for text in texts]

# Create embedding and vector store
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)

# Match controls by cosine similarity
def match_controls(query, threshold=0.75):
    results = vectorstore.similarity_search_with_score(query, k=10)
    return [(doc, score) for doc, score in results if score >= threshold]

# Run matching loop
if __name__ == "__main__":
    query = input("Enter a company policy line: ")
    matches = match_controls(query)

    if not matches:
        print("\n❌ No relevant control matches found above threshold.\n")
    else:
        print(f"\n✅ Matched Controls (Score ≥ {0.75}):\n")
        for i, (doc, score) in enumerate(matches, 1):
            print(f"{i}. {doc.page_content}  (Score: {score:.4f})")
