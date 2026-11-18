
import json
import re

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------- CONFIGURATION ----------------

KB_FILE = "knowledge_base.json"

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

word_vectorizer = None
char_vectorizer = None

passage_word_matrix = None
passage_char_matrix = None

sentence_word_matrix = None
sentence_char_matrix = None


# ---------------- REQUEST FORMAT ----------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=5)


# ---------------- LOAD KNOWLEDGE BASE ----------------

@app.on_event("startup")
def load_data():

    global passages, sentences, sentence_sources
    global word_vectorizer, char_vectorizer
    global passage_word_matrix, passage_char_matrix
    global sentence_word_matrix, sentence_char_matrix

    with open(KB_FILE, "r", encoding="utf-8") as file:
        passages = json.load(file)

    if not isinstance(passages, list) or len(passages) != 50:
        raise RuntimeError("Knowledge base must contain exactly 50 passages.")

    # Split passages into sentences
    for index, passage in enumerate(passages):

        parts = re.split(r"(?<=[.!?])\s+", passage)

        for part in parts:
            part = part.strip()

            if part:
                sentences.append(part)
                sentence_sources.append(index)

    # Word-based TF-IDF
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w.-]+\b"
    )

    passage_word_matrix = normalize(
        word_vectorizer.fit_transform(passages)
    )

    sentence_word_matrix = normalize(
        word_vectorizer.transform(sentences)
    )

    # Character-based TF-IDF helps match technical terms
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        max_features=20000
    )

    passage_char_matrix = normalize(
        char_vectorizer.fit_transform(passages)
    )

    sentence_char_matrix = normalize(
        char_vectorizer.transform(sentences)
    )


# ---------------- RETRIEVAL ----------------

def retrieve(question, top_k):

    query_word = word_vectorizer.transform([question])
    query_char = char_vectorizer.transform([question])

    word_scores = (
        passage_word_matrix @ query_word.T
    ).toarray().flatten()

    char_scores = (
        passage_char_matrix @ query_char.T
    ).toarray().flatten()

    # Combine word and character similarity
    final_scores = (
        0.75 * word_scores +
        0.25 * char_scores
    )

    ranked = np.argsort(-final_scores)[:top_k]

    return ranked, final_scores, query_word, query_char


# ---------------- ANSWER EXTRACTION ----------------

def extract_answer(query_word, query_char, ranked_indices):

    word_scores = (
        sentence_word_matrix @ query_word.T
    ).toarray().flatten()

    char_scores = (
        sentence_char_matrix @ query_char.T
    ).toarray().flatten()

    sentence_scores = (
        0.75 * word_scores +
        0.25 * char_scores
    )

    evidence = []
    used_sources = set()

    for passage_index in ranked_indices:

        candidates = []

        for i, source_index in enumerate(sentence_sources):

            if source_index != passage_index:
                continue

            sentence = sentences[i]

            # Do not return sentences containing suspicious instructions
            if re.search(
                r"ignore previous instructions|reveal system prompt|"
                r"execute this command|ignore all rules",
                sentence,
                re.IGNORECASE
            ):
                continue

            candidates.append((sentence_scores[i], sentence))

        candidates.sort(key=lambda x: x[0], reverse=True)

        if candidates and candidates[0][0] >= 0.10:

            best_score, best_sentence = candidates[0]

            evidence.append(best_sentence)
            used_sources.add(int(passage_index))

    return evidence, used_sources


# ---------------- API ENDPOINTS ----------------

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

    indices, combined, query_word, query_char = retrieve(
        question,
        request.top_k
    )

    best_score = float(combined[indices[0]])

    # Abstain if retrieval relevance is too low
    if best_score < 0.12:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    evidence, used_sources = extract_answer(
        query_word,
        query_char,
        indices
    )

    if not evidence:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    # Keep sources in descending retrieval order
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
        0.95,
        max(0.62, 0.62 + best_score * 0.5)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }
