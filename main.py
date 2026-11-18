
import json
import re
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------------- CONFIGURATION ----------------

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


# ---------------- TECHNICAL ENTITIES ----------------

ENTITY_ALIASES = {
    "postgresql": ["postgresql", "postgres", "pgsql"],
    "redis": ["redis"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "cassandra": ["cassandra"],

    "raft": ["raft"],
    "paxos": ["paxos", "multi-paxos", "multipaxos"],

    "sgd": ["sgd", "stochastic gradient descent"],
    "adamw": ["adamw", "adam w"],
    "adam": ["adam optimizer"],
    "layer normalization": ["layer normalization", "layernorm"],

    "mvcc": ["mvcc", "multi-version concurrency control"],
    "undo logs": ["undo log", "undo logs"],

    "tcp": ["tcp", "transmission control protocol"],
    "udp": ["udp", "user datagram protocol"],
    "dns": ["dns", "domain name system"],
    "bgp": ["bgp", "border gateway protocol"],
    "ospf": ["ospf"],

    "tls": ["tls", "transport layer security"],
    "ssl": ["ssl"],
    "rsa": ["rsa"],
    "aes": ["aes"],
    "sha": ["sha", "sha-256", "sha256"],
}


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def contains_phrase(text, phrase):
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text.lower()))


def find_entities(text):
    found = []

    for entity, aliases in ENTITY_ALIASES.items():
        if any(contains_phrase(text, alias) for alias in aliases):
            found.append(entity)

    return found


# ---------------- PROMPT INJECTION FILTER ----------------

def is_injection(text):
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+all\s+rules",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"execute\s+this\s+command",
        r"disregard\s+(all\s+)?instructions",
        r"override\s+(the\s+)?system",
        r"ignore\s+the\s+above",
    ]

    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


# ---------------- LOAD KNOWLEDGE BASE ----------------

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

    sentences.clear()
    sentence_sources.clear()

    for index, passage in enumerate(passages):

        parts = re.split(r"(?<=[.!?])\s+", passage)

        for part in parts:

            part = clean_text(part)

            if part and not is_injection(part):
                sentences.append(part)
                sentence_sources.append(index)

    # Dense semantic embeddings
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

    # Lexical retrieval
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


# ---------------- RETRIEVAL ----------------

def retrieve(question, top_k):

    query_embedding = np.array(
        list(embedder.query_embed(question)),
        dtype=np.float32
    )

    query_embedding = normalize(query_embedding)

    query_sparse = vectorizer.transform([question])

    dense_scores = (
        passage_embeddings @ query_embedding.T
    ).flatten()

    lexical_scores = (
        passage_tfidf @ query_sparse.T
    ).toarray().flatten()

    # Hybrid retrieval
    scores = (
        0.70 * dense_scores +
        0.30 * lexical_scores
    )

    query_entities = find_entities(question)

    # Entity-aware relevance boost
    if query_entities:

        for i, passage in enumerate(passages):

            passage_entities = find_entities(passage)

            matching = set(query_entities) & set(passage_entities)

            if matching:
                scores[i] += 0.12 * (
                    len(matching) / len(query_entities)
                )

    scores = np.clip(scores, 0, 1)

    # For comparisons, retrieve enough passages to find both sides
    number_to_retrieve = min(
        len(passages),
        max(top_k, 10 if len(query_entities) >= 2 else 5)
    )

    ranked = np.argsort(-scores)[:number_to_retrieve]

    return ranked, scores, query_embedding, query_sparse, query_entities


# ---------------- EVIDENCE EXTRACTION ----------------

def extract_evidence(
    query_embedding,
    query_sparse,
    ranked,
    query_entities
):

    dense_scores = (
        sentence_embeddings @ query_embedding.T
    ).flatten()

    lexical_scores = (
        sentence_tfidf @ query_sparse.T
    ).toarray().flatten()

    sentence_scores = (
        0.70 * dense_scores +
        0.30 * lexical_scores
    )

    evidence = []
    selected_sources = []
    used_sentences = set()

    for passage_index in ranked:

        passage = passages[int(passage_index)]
        passage_entities = find_entities(passage)

        matching_entities = (
            set(query_entities) & set(passage_entities)
        )

        # If question names specific entities, avoid unrelated passages
        if query_entities and not matching_entities:
            continue

        candidates = []

        for i, source in enumerate(sentence_sources):

            if source != passage_index:
                continue

            sentence = sentences[i]

            if is_injection(sentence):
                continue

            score = float(sentence_scores[i])

            sentence_entities = find_entities(sentence)

            # Reward sentences that mention the requested entities
            if query_entities:

                sentence_matches = (
                    set(query_entities) & set(sentence_entities)
                )

                score += 0.12 * (
                    len(sentence_matches) / len(query_entities)
                )

                # Keep sentence context from a passage
                # that contains the requested entity.
                if not sentence_matches and not matching_entities:
                    continue

            candidates.append((score, i, sentence))

        candidates.sort(key=lambda x: x[0], reverse=True)

        # Select up to two useful sentences per passage
        count = 0

        for score, sentence_index, sentence in candidates:

            if count >= 2:
                break

            if score < 0.25:
                continue

            if sentence_index in used_sentences:
                continue

            evidence.append(sentence)
            selected_sources.append(int(passage_index))
            used_sentences.add(sentence_index)

            count += 1

    return evidence, selected_sources


# ---------------- API ENDPOINTS ----------------

@app.get("/")
def home():
    return {
        "status": "RAG API is running"
    }


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

    ranked, scores, query_embedding, query_sparse, query_entities = (
        retrieve(question, request.top_k)
    )

    best_score = float(scores[ranked[0]])

    # Abstain on weak retrieval
    if best_score < 0.35:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    evidence, evidence_sources = extract_evidence(
        query_embedding,
        query_sparse,
        ranked,
        query_entities
    )

    if not evidence:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    # Keep source indices unique and in descending retrieval order
    source_set = set(evidence_sources)

    sources = [
        int(i) for i in ranked
        if int(i) in source_set
    ]

    # Preserve evidence order while avoiding duplicate sentences
    answer = " ".join(dict.fromkeys(evidence))

    output_scores = [
        round(float(np.clip(scores[i], 0, 1)), 4)
        for i in sources
    ]

    # Confidence based on retrieval strength
    confidence = min(
        0.95,
        max(0.62, 0.62 + best_score * 0.30)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": output_scores,
        "confidence": round(confidence, 4)
    }
