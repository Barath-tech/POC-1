import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from llm import gpt_category_name, gpt_summarize
from memory import add_category

def embed_texts(texts, model):
    print("🧠 Generating embeddings...")
    return model.encode(texts, batch_size=32, show_progress_bar=True)

def cluster_and_create_categories(df, embeddings, st_model, mem, n_clusters=8):
    print(f"📊 Clustering into {n_clusters} categories...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    df["_cluster"] = labels

    for cl in set(labels):
        members = df[df["_cluster"] == cl]
        sample_text = members.iloc[0]["_text"]
        print(f"🔸 Creating initial category for cluster {cl}")
        name = gpt_category_name(sample_text)
        add_category(mem, st_model, name, sample_text)
    return df

def match_to_categories(emb, mem):
    if not mem["categories"]:
        print("⚠️ No categories in memory yet.")
        return None, 0.0

    mem_names = list(mem["categories"].keys())
    mem_embs = [v["embedding"] for v in mem["categories"].values()]

    sims = cosine_similarity([emb], mem_embs)[0]
    idx = sims.argmax()
    print(f"🔍 Matched with category '{mem_names[idx]}' (score={sims[idx]:.3f})")
    return mem_names[idx], sims[idx]
