import pandas as pd
import requests
import random
import numpy as np
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load Data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/mohammedifran13/chatbot-zeotap/combined_scraped_data.csv")  # Ensure this file is in your project root

# BM25 Retrieval
def get_bm25():
    df = load_data()
    if "Scraped_Text" not in df.columns:
        raise ValueError("CSV file is missing the 'Scraped_Text' column.")
    
    tokenized_corpus = [str(doc).split() for doc in df["Scraped_Text"].astype(str).str.lower()]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25, df

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini API Keys
GEMINI_API_KEYS = [
    "AIzaSyArR5jH7vDbmKR1dyXaRQ3thCZZOqOtE5U",
    "AIzaSyDhzilC3QWrI10xHztZX3mjE3LHZ4qoXI4",
    "AIzaSyCXSSi-499C_ifRNT_ZUCf0l5MOuakX5XM"
]

def set_gemini_api_key():
    api_key = random.choice(GEMINI_API_KEYS)
    genai.configure(api_key=api_key)

# Search Documents
def search_documents(query, top_n=5):
    bm25, df = get_bm25()
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    if len(scores) == 0:
        return pd.DataFrame(columns=["URL", "Scraped_Text"])
    
    top_indices = np.argsort(scores)[-top_n:][::-1]
    retrieved_docs = df.iloc[top_indices].copy()
    
    if "URL" not in retrieved_docs.columns:
        retrieved_docs["URL"] = "Unknown"
    
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    retrieved_docs["Similarity"] = retrieved_docs["Scraped_Text"].apply(
        lambda x: util.pytorch_cos_sim(query_embedding, embedding_model.encode(str(x), convert_to_tensor=True)).item()
    )
    
    return retrieved_docs.sort_values(by="Similarity", ascending=False).head(top_n)

# Extract Answer
def extract_answer(text, query):
    text = text.lower()
    query = query.lower()
    return text if query in text else " ".join(text.split()[:300])

# Refine with Gemini
def refine_with_gemini(query, text, url, retry_count=0):
    if not text or "⚠️" in text:
        return "⚠️ No relevant content to refine."
    set_gemini_api_key()
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are an AI assistant. Summarize the given content clearly.
    
    **Query:** {query}
    **Extracted Answer:** {text}
    
    **Refined Answer (Source: {url})**:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        if retry_count < len(GEMINI_API_KEYS):
            return refine_with_gemini(query, text, url, retry_count + 1)
        return "⚠️ Gemini API unavailable."

# Further Refinement
def further_refine_with_gemini(query, initial_output, retry_count=0):
    if not initial_output or "⚠️" in initial_output:
        return "⚠️ No meaningful content to refine."
    set_gemini_api_key()
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are refining an AI-generated response. Ensure clarity and structure.
    
    **Query:** {query}
    
    **Initial Answer:**
    {initial_output}
    
    **Final Refined Answer:**
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        if retry_count < len(GEMINI_API_KEYS):
            return further_refine_with_gemini(query, initial_output, retry_count + 1)
        return "⚠️ Gemini API unavailable."

# External Data Fetching
def fetch_external_data(query):
    api_endpoints = [
        "https://phidata-flask.onrender.com/query",
        "https://phidata-flask-9qv9.onrender.com/query"
    ]
    
    enhanced_query = f"""
    {query}
    
    **Instructions:**
    - Remove irrelevant text.
    - Combine key points into a structured, clear answer.
    - Keep only **one** source URL (the most relevant one).
    """
    
    payload = {"question": enhanced_query}
    headers = {"Content-Type": "application/json"}
    
    for url in api_endpoints:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json().get("response", "")
            return "\n".join(set(data.split("\n"))).strip()
        except requests.RequestException:
            continue
    return "⚠️ Try again"

# Answer Query
def answer_query(query):
    query_lower = query.lower()

    # 🔹 Ensure the query specifies a CDP
    if not any(cdp in query_lower for cdp in ["zeotap", "segment", "lytics", "mparticle"]):
        return "⚠️ Specify which CDP you need: Segment, mParticle, Zeotap, or Lytics."

    if "zeotap" in query_lower or "segment" in query_lower:
        return fetch_external_data(query)

    if "lytics" in query_lower or "mparticle" in query_lower:
        relevant_docs = search_documents(query, top_n=10)

        if relevant_docs.empty:
            return fetch_external_data(query)  # Fallback if no relevant documents are found

        all_refined_answers = []
        for _, row in relevant_docs.iterrows():
            extracted_answer = extract_answer(row["Scraped_Text"], query)

            if not extracted_answer or "⚠️" in extracted_answer:
                continue  # Skip empty or irrelevant extracted answers

            refined_answer = refine_with_gemini(query, extracted_answer, row["URL"])

            if "⚠️" not in refined_answer and refined_answer.strip():
                all_refined_answers.append(refined_answer)

        if not all_refined_answers:
            return fetch_external_data(query)  # Fallback if no meaningful content was refined

        combined_answer = "\n\n".join(all_refined_answers)
        final_answer = further_refine_with_gemini(query, combined_answer)

        # 🔹 Check if the output contains specific fallback phrases
        fallback_phrases = [
            "the provided text does not contain",
            "please consult the official documentation."
        ]

        if any(phrase in final_answer.lower() for phrase in fallback_phrases):
            return fetch_external_data(query)

        return final_answer

    return fetch_external_data(query)  # Default fallback

# Flask API Route
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("question", "")
    if not query_text:
        return jsonify({"response": "⚠️ No query provided."})

    response = answer_query(query_text)
    return jsonify({"response": response})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get PORT from Render
    app.run(host="0.0.0.0", port=port)  # Bind to all interfaces

