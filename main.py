
import json
import re
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

KB_FILE = "knowledge_base.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Technical RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

passages = []
sentences = []
sentence_sources = []

embedder = None
passage_embeddings = None
sentence_embeddings = None

vectorizer = None
passage_tfidf = None
sentence_tfidf = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=5)


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def is_injection(text):
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+all\s+rules",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"execute\s+this\s+command",
        r"disregard\s+(all\s+)?instructions",
    ]

    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


@app.on_event("startup")
def load_data():
    global passages, sentences, sentence_sources
    global embedder, passage_embeddings, sentence_embeddings
    global vectorizer, passage_tfidf, sentence_tfidf

    with open(KB_FILE, "r", encoding="utf-8") as file:
        passages = json.load(file)

    if not isinstance(passages, list) or len(passages) != 50:
        raise RuntimeError("Knowledge base must contain exactly 50 passages.")

    passages = [clean_text(p) for p in passages]

    for index, passage in enumerate(passages):
        parts = re.split(r"(?<=[.!?])\s+", passage)

        for part in parts:
            part = clean_text(part)

            if part and not is_injection(part):
                sentences.append(part)
                sentence_sources.append(index)

    # Load dense embedding model
    embedder = TextEmbedding(model_name=MODEL_NAME)

    passage_embeddings = np.array(
        list(embedder.embed(passages)),
        dtype=np.float32
    )

    sentence_embeddings = np.array(
        list(embedder.embed(sentences)),
        dtype=np.float32
    )

    passage_embeddings = normalize(passage_embeddings)
    sentence_embeddings = normalize(sentence_embeddings)

    # Keep TF-IDF for exact technical terms
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w.-]+\b",
        max_features=20000
    )

    passage_tfidf = normalize(
        vectorizer.fit_transform(passages)
    )

    sentence_tfidf = normalize(
        vectorizer.transform(sentences)
    )


def retrieve(question, top_k):
    query_embedding = np.array(
        list(embedder.query_embed(question)),
        dtype=np.float32
    )

    query_embedding = normalize(query_embedding)

    dense_scores = (
        passage_embeddings @ query_embedding.T
    ).flatten()

    query_sparse = vectorizer.transform([question])

    lexical_scores = (
        passage_tfidf @ query_sparse.T
    ).toarray().flatten()

    scores = (
        0.75 * dense_scores +
        0.25 * lexical_scores
    )

    ranked = np.argsort(-scores)[:top_k]

    return ranked, scores, query_embedding, query_sparse


def extract_evidence(query_embedding, query_sparse, ranked):
    dense_sentence_scores = (
        sentence_embeddings @ query_embedding.T
    ).flatten()

    lexical_sentence_scores = (
        sentence_tfidf @ query_sparse.T
    ).toarray().flatten()

    sentence_scores = (
        0.75 * dense_sentence_scores +
        0.25 * lexical_sentence_scores
    )

    evidence = []
    used_sources = set()

    for passage_index in ranked:
        candidates = []

        for i, source in enumerate(sentence_sources):
            if source != passage_index:
                continue

            sentence = sentences[i]

            if is_injection(sentence):
                continue

            candidates.append((sentence_scores[i], sentence))

        candidates.sort(key=lambda x: x[0], reverse=True)

        if candidates and candidates[0][0] >= 0.30:
            evidence.append(candidates[0][1])
            used_sources.add(int(passage_index))

    return evidence, used_sources


@app.get("/")
def home():
    return {"status": "RAG API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "passages": len(passages),
        "embedding_model": MODEL_NAME
    }


@app.post("/query")
def query(request: QueryRequest):
    question = clean_text(request.question)

    if not question or is_injection(question):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    ranked, scores, query_embedding, query_sparse = retrieve(
        question,
        request.top_k
    )

    best_score = float(scores[ranked[0]])

    # Conservative abstention
    if best_score < 0.35:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    evidence, used_sources = extract_evidence(
        query_embedding,
        query_sparse,
        ranked
    )

    if not evidence:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    sources = [
        int(i) for i in ranked
        if int(i) in used_sources
    ]

    answer = " ".join(evidence)

    output_scores = [
        round(float(np.clip(scores[i], 0, 1)), 4)
        for i in sources
    ]

    confidence = min(
        0.95,
        max(0.62, 0.62 + best_score * 0.35)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": output_scores,
        "confidence": round(confidence, 4)
    }
