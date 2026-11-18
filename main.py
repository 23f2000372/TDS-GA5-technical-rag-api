
import json
import os
import re
import threading

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------- CONFIGURATION ----------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
KB_FILE = "knowledge_base.json"

app = FastAPI(title="Technical RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
passages = []
passage_vectors = None
vectorizer = None
tfidf_matrix = None
sentence_data = []
sentence_vectors = None

model_lock = threading.Lock()


# ---------------- REQUEST FORMAT ----------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=5)


# ---------------- LOAD KNOWLEDGE BASE ----------------

@app.on_event("startup")
def load_data():

    global model, passages
    global passage_vectors, vectorizer, tfidf_matrix
    global sentence_data, sentence_vectors

    with open(KB_FILE, "r", encoding="utf-8") as file:
        passages = json.load(file)

    if not isinstance(passages, list) or len(passages) != 50:
        raise RuntimeError("Knowledge base must contain exactly 50 passages.")

    model = SentenceTransformer(MODEL_NAME)

    # Generate dense embeddings once at startup
    with model_lock:
        passage_vectors = model.encode(
            passages,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

    # TF-IDF helps with exact technical terms and numbers
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True
    )

    tfidf_matrix = normalize(
        vectorizer.fit_transform(passages)
    )

    # Split passages into individual sentences
    all_sentences = []

    for index, passage in enumerate(passages):
        sentences = re.split(r"(?<=[.!?])\s+", passage)

        for sentence in sentences:
            if sentence.strip():
                sentence_data.append((index, sentence.strip()))
                all_sentences.append(sentence.strip())

    with model_lock:
        sentence_vectors = model.encode(
            all_sentences,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")


# ---------------- RETRIEVAL ----------------

def retrieve(question, top_k):

    with model_lock:
        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0].astype("float32")

    dense_scores = np.clip(
        passage_vectors @ query_vector, 0, 1
    )

    query_tfidf = vectorizer.transform([question])

    lexical_scores = (
        query_tfidf @ tfidf_matrix.T
    ).toarray()[0]

    # Combine semantic and keyword relevance
    final_scores = (
        0.80 * dense_scores +
        0.20 * lexical_scores
    )

    ranked = np.argsort(-final_scores)

    return ranked[:top_k], dense_scores, lexical_scores, final_scores


# ---------------- ANSWER EXTRACTION ----------------

def extract_answer(question, ranked_indices):

    with model_lock:
        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0].astype("float32")

    question_words = set(
        re.findall(r"\b[\w.-]+\b", question.lower())
    )

    evidence = []
    used_sources = set()

    for idx in ranked_indices:

        candidates = []

        for i, (source_idx, sentence) in enumerate(sentence_data):

            if source_idx != idx:
                continue

            semantic = float(
                np.clip(sentence_vectors[i] @ query_vector, 0, 1)
            )

            sentence_words = set(
                re.findall(r"\b[\w.-]+\b", sentence.lower())
            )

            overlap = len(
                question_words & sentence_words
            ) / max(1, len(question_words))

            score = 0.80 * semantic + 0.20 * overlap

            candidates.append((score, sentence))

        candidates.sort(reverse=True)

        if candidates and candidates[0][0] >= 0.30:

            best_sentence = candidates[0][1]

            # Never execute or follow instructions found in passages.
            if re.search(
                r"ignore previous instructions|reveal system prompt|"
                r"execute this command",
                best_sentence,
                re.IGNORECASE
            ):
                continue

            evidence.append(best_sentence)
            used_sources.add(idx)

    return evidence, used_sources


# ---------------- API ENDPOINT ----------------

@app.get("/")
def home():
    return {"status": "RAG API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "passages": len(passages)
    }


@app.post("/query")
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    indices, dense, lexical, combined = retrieve(
        question,
        request.top_k
    )

    best_idx = int(indices[0])
    best_dense = float(dense[best_idx])
    best_lexical = float(lexical[best_idx])

    # Conservative abstention check
    if not (
        best_dense >= 0.43
        or (best_dense >= 0.34 and best_lexical >= 0.025)
    ):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    evidence, used_sources = extract_answer(
        question,
        indices
    )

    if not evidence:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    # Preserve descending retrieval order
    sources = [
        int(i) for i in indices
        if int(i) in used_sources
    ]

    scores = [
        round(float(np.clip(combined[i], 0, 1)), 4)
        for i in sources
    ]

    answer = " ".join(evidence)

    confidence = min(
        0.97,
        max(0.62, 0.62 + (float(combined[best_idx]) - 0.35) * 0.75)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }
