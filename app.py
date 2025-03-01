from flask import Flask, request, jsonify
import pandas as pd
import nltk
import spacy
import requests
import random
from google import genai
from nltk.corpus import stopwords, wordnet
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# Initialize Flask App
app = Flask(__name__)

# 🔹 Download necessary NLP models
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Load CSV Data
df = pd.read_csv("combined_scraped_data.csv")
df["Processed_Text"] = df["Scraped_Text"].astype(str).str.lower()

# 🔹 Prepare BM25 for search
tokenized_corpus = [doc.split() for doc in df["Processed_Text"]]
bm25 = BM25Okapi(tokenized_corpus)

# 🔹 Gemini API Key Rotation
GEMINI_API_KEYS = [
    "AIzaSyArR5jH7vDbmKR1dyXaRQ3thCZZOqOtE5U",
    "AIzaSyDhzilC3QWrI10xHztZX3mjE3LHZ4qoXI4",
    "AIzaSyCXSSi-499C_ifRNT_ZUCf0l5MOuakX5XM"
]

def get_gemini_client():
    """Randomly selects a Gemini API key to avoid rate limits."""
    api_key = random.choice(GEMINI_API_KEYS)
    return genai.Client(api_key=api_key)

# 🔹 Query Expansion with Synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def expand_query(query):
    words = query.split()
    expanded_words = [word for word in words]
    for word in words:
        expanded_words.extend(get_synonyms(word))
    return " ".join(expanded_words)

# 🔹 Document Search using BM25 and Semantic Similarity
def search_documents(query, top_n=10):
    expanded_query = expand_query(query)
    query_tokens = expanded_query.split()
    scores = bm25.get_scores(query_tokens)

    # Get top N documents based on BM25 scores
    top_indices = scores.argsort()[-top_n:][::-1]
    retrieved_docs = df.iloc[top_indices][["URL", "Source", "Scraped_Text"]].copy()

    # 🔹 Re-rank using Semantic Similarity
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    retrieved_docs["Similarity"] = retrieved_docs["Scraped_Text"].apply(
        lambda x: util.pytorch_cos_sim(query_embedding, embedding_model.encode(x, convert_to_tensor=True)).item()
    )

    # Sort by Semantic Similarity
    retrieved_docs = retrieved_docs.sort_values(by="Similarity", ascending=False)

    # Keep only highly relevant snippets (Similarity > 0.5)
    retrieved_docs = retrieved_docs[retrieved_docs["Similarity"] > 0.5]

    return retrieved_docs

# 🔹 Extract the Most Relevant Answer Sentence
def extract_answer(text, query):
    doc = nlp(text)
    query_keywords = set(query.lower().split())

    best_sentence = None
    max_match = 0

    for sent in doc.sents:
        words = set(sent.text.lower().split())
        match_count = len(words & query_keywords)

        if match_count > max_match:
            best_sentence = sent.text
            max_match = match_count

    return best_sentence if best_sentence else "⚠️ No relevant sentence found."

# 🔹 Answer Query API Endpoint
@app.route('/query', methods=['POST'])
def answer_query():
    data = request.get_json()
    query = data.get("question", "")

    if not query:
        return jsonify({"error": "⚠️ Query cannot be empty."}), 400

    # 🔹 Fetch relevant documents
    relevant_docs = search_documents(query, top_n=10)

    if relevant_docs.empty:
        return jsonify({"response": "⚠️ No relevant documents found."})

    all_refined_answers = []

    for _, row in relevant_docs.iterrows():
        extracted_answer = extract_answer(row["Scraped_Text"], query)
        all_refined_answers.append(f"Source: {row['URL']}\n{extracted_answer}")

    combined_answer = "\n\n".join(all_refined_answers)
    
    return jsonify({"response": combined_answer})

# 🔹 Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
