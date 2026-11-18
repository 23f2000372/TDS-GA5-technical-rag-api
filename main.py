
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


# ---------------- ENTITY ALIASES ----------------

ENTITY_GROUPS = {
    "raft": ["raft"],
    "paxos": ["paxos", "multi-paxos"],
    "gossip": ["gossip", "epidemic gossip"],
    "consistent hashing": ["consistent hashing"],
    "vector clocks": ["vector clock", "vector clocks"],
    "sgd": ["sgd", "stochastic gradient descent"],
    "adamw": ["adamw", "adam w"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb"],
    "cassandra": ["cassandra", "apache cassandra", "cql"],
    "redis": ["redis"],
    "tcp": ["tcp", "transmission control protocol"],
    "quic": ["quic"],
    "dns": ["dns", "domain name system"],
    "bgp": ["bgp", "border gateway protocol"],
    "ospf": ["ospf", "open shortest path first"],
    "ecc": ["elliptic curve cryptography", "ecdsa"],
}


def normalize_text(text):
    text = text.lower()
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def find_entities(text):
    text = normalize_text(text)
    found = []

    for entity, aliases in ENTITY_GROUPS.items():
        for alias in aliases:
            alias = normalize_text(alias)

            if re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                text
            ):
                found.append(entity)
                break

    return found


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

    sentences = []
    sentence_sources = []

    for index, passage in enumerate(passages):

        parts = re.split(r"(?<=[.!?])\s+", passage)

        for part in parts:
            part = part.strip()

            if part:
                sentences.append(part)
                sentence_sources.append(index)

    # Word TF-IDF

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        stop_words="english",
        token_pattern=r"(?u)\b[\w.-]+\b"
    )

    passage_word_matrix = normalize(
        word_vectorizer.fit_transform(passages)
    )

    sentence_word_matrix = normalize(
        word_vectorizer.transform(sentences)
    )

    # Character TF-IDF

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

    scores = (
        0.70 * word_scores +
        0.30 * char_scores
    )

    query_entities = find_entities(question)

    # Entity boost, without forcing passage order

    if query_entities:

        for i, passage in enumerate(passages):

            passage_entities = set(find_entities(passage))
            matching = set(query_entities) & passage_entities

            if matching:
                scores[i] += 0.15 * (
                    len(matching) / len(query_entities)
                )

    # Always rank by final score

    ranked = np.argsort(-scores)

    return (
        ranked[:max(top_k, len(query_entities))],
        scores,
        query_word,
        query_char,
        query_entities
    )


# ---------------- ANSWER EXTRACTION ----------------

INJECTION_PATTERN = re.compile(
    r"ignore previous instructions|"
    r"reveal system prompt|"
    r"execute this command|"
    r"ignore all rules|"
    r"disregard previous instructions",
    re.IGNORECASE
)



def extract_answer(
    question,
    query_word,
    query_char,
    ranked_indices,
    query_entities
):

    word_scores = (
        sentence_word_matrix @ query_word.T
    ).toarray().flatten()

    char_scores = (
        sentence_char_matrix @ query_char.T
    ).toarray().flatten()

    sentence_scores = (
        0.70 * word_scores +
        0.30 * char_scores
    )

    question_words = set(
        re.findall(r"[a-z0-9]+", normalize_text(question))
    )

    evidence_pairs = []
    used_sources = []

    for passage_index in ranked_indices:

        passage_index = int(passage_index)
        passage = passages[passage_index]

        passage_entities = set(find_entities(passage))
        matching_entities = set(query_entities) & passage_entities

        if query_entities and not matching_entities:
            continue

        candidates = []

        for i, source_index in enumerate(sentence_sources):

            if source_index != passage_index:
                continue

            sentence = sentences[i]

            if INJECTION_PATTERN.search(sentence):
                continue

            sentence_words = set(
                re.findall(r"[a-z0-9]+", normalize_text(sentence))
            )

            overlap = len(question_words & sentence_words)

            score = float(sentence_scores[i])

            # Prefer sentences matching the question

            score += min(overlap * 0.02, 0.15)

            # Passage context already identifies the entity.
            # Do not require every sentence to repeat its name.

            if query_entities:
                sentence_entities = set(find_entities(sentence))

                if set(query_entities) & sentence_entities:
                    score += 0.10

            candidates.append((score, sentence))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []

        for score, sentence in candidates:

            if score < 0.08:
                continue

            if sentence not in selected:
                selected.append(sentence)

            if len(selected) >= 2:
                break

        if selected:

            for sentence in selected:
                evidence_pairs.append(
                    (sentence, passage_index)
                )

            used_sources.append(passage_index)

    return evidence_pairs, used_sources



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

from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    

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

    indices, combined, query_word, query_char, query_entities = retrieve(
        question,
        request.top_k
    )

    best_score = float(combined[indices[0]])

    if best_score < 0.10:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    evidence, used_sources = extract_answer(
        question,
        query_word,
        query_char,
        indices,
        query_entities
    )

    if not evidence or not used_sources:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    # Rank sources by their actual retrieval scores

    sources = sorted(
        set(used_sources),
        key=lambda i: combined[i],
        reverse=True
    )[:request.top_k]

    scores = [
        round(float(np.clip(combined[i], 0, 1)), 4)
        for i in sources
    ]

    # Keep evidence and its original source together

    allowed_sources = set(sources)

    final_evidence = []

    for sentence, source_index in evidence:

        if source_index in allowed_sources:
            if sentence not in final_evidence:
                final_evidence.append(sentence)

    answer = " ".join(final_evidence)

    if not answer:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    confidence = min(
        0.95,
        max(0.40, 0.45 + best_score * 0.65)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }
