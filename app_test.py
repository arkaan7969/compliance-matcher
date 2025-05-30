import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 1. Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. Load controls.csv
df = pd.read_csv("controls_nist.csv")
controls = df.to_dict(orient="records")
texts = [f"{c['id']} ({c['framework']}): {c['description']}" for c in controls]

# 3. Initialize embedding model
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. Embed all control texts once
control_vecs = np.array(embedder.embed_documents(texts))
control_norms = np.linalg.norm(control_vecs, axis=1)

# 5. Matching function
def get_top_matches(query: str, top_n: int = 3, threshold: float = 0.4):  # lowered threshold
    q_vec = np.array(embedder.embed_query(query))
    q_norm = np.linalg.norm(q_vec)
    sims = (control_vecs @ q_vec) / (control_norms * q_norm + 1e-12)
    valid = [(i, s) for i, s in enumerate(sims) if s >= threshold]
    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[:top_n]

# 6. Run test cases
def run_test_cases(file_path: str):
    with open(file_path, "r") as f:
        test_cases = [line.strip() for line in f if line.strip()]

    for idx, case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {idx}: {case}")
        matches = get_top_matches(case)
        if not matches:
            print("❌ No relevant control matches found.")
        else:
            print("✅ Matches:")
            for rank, (i, score) in enumerate(matches, 1):
                print(f"{rank}. {texts[i]}  (Score: {score:.4f})")

# 7. Entry point
if __name__ == "__main__":
    run_test_cases("test_cases.txt")
