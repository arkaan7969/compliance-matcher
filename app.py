import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI

# 1. Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. Load controls from CSV
df = pd.read_csv("controls_nist.csv")
controls = df.to_dict(orient="records")
texts = [f"{c['id']} ({c['framework']}): {c['description']}" for c in controls]
docs = [Document(page_content=text) for text in texts]

# 3. Create or load Chroma vectorstore with OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Persist directory for Chroma DB
persist_dir = "./chroma_db"

# Check if DB exists, else create
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()

# 4. Matching function
def match_policy(policy_input):
    return vectorstore.similarity_search(policy_input, k=3)

# 5. Explanation function using GPT-3.5-turbo
def explain_matches(policy_input, matched_docs):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    context = "\n".join([doc.page_content for doc in matched_docs])
    prompt = f"""
You are a cybersecurity compliance expert.

A company policy states: "{policy_input}"

Below are related compliance controls:

{context}

For each control, provide a concise explanation including:
- Control ID
- Why it matches the policy
- Suggested remediation steps or actions to align with the control
- How to collect evidence for compliance

Additionally, review the policy statement itself and provide your professional opinion on whether the policy is adequate, or if you recommend any improvements or cautions, especially around timelines, feasibility, or best practices.

Answer clearly and directly, with a personal touch as if advising a company’s security team.
"""
    return llm.predict(prompt)

# 6. Main runner
if __name__ == "__main__":
    policy_input = input("Enter company policy: ")
    matched_docs = match_policy(policy_input)
    explanation = explain_matches(policy_input, matched_docs)
    print("\n--- Matched Controls Explanation ---\n")
    print(explanation)
