
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
    "raft": ["raft", "etcd"],
    "paxos": ["paxos", "multi-paxos", "zookeeper", "chubby"],
    "gossip": ["gossip", "epidemic gossip", "cassandra gossip"],
    "swim": ["swim", "consul"],
    "two phase commit": ["two-phase commit", "2pc", "two phase commit"],
    "three phase commit": ["three-phase commit", "3pc", "three phase commit"],
    "saga": ["saga", "cadence"],
    "consistent hashing": ["consistent hashing", "dynamo"],
    "rendezvous hashing": ["rendezvous hashing", "highest random weight"],
    "vector clocks": ["vector clock", "vector clocks", "riak"],
    "hybrid logical clocks": ["hybrid logical clock", "hlc"],
    "sgd": ["sgd", "stochastic gradient descent"],
    "adamw": ["adamw", "adam w"],
    "transformer encoder": ["transformer encoder", "self-attention"],
    "transformer decoder": ["transformer decoder", "autoregressive decoder", "llama"],
    "logistic regression": ["logistic regression"],
    "svm": ["svm", "support vector machine"],
    "mlp": ["mlp", "multi-layer perceptron", "multilayer perceptron"],
    "resnet": ["resnet", "residual convolutional network"],
    "batch normalization": ["batch normalization", "batch norm"],
    "layer normalization": ["layer normalization", "layer norm"],
    "postgresql": ["postgresql", "postgres", "pgbouncer"],
    "mysql": ["mysql", "innodb"],
    "b tree": ["b+ tree", "b tree", "clustered index"],
    "lsm tree": ["lsm tree", "log-structured merge tree", "rocksdb"],
    "mongodb": ["mongodb", "wiredtiger"],
    "cassandra": ["apache cassandra", "cassandra", "cql"],
    "redis": ["redis"],
    "memcached": ["memcached"],
    "wal": ["write-ahead log", "write ahead log", "wal", "aries"],
    "shadow paging": ["shadow paging", "sqlite"],
    "tcp": ["tcp", "transmission control protocol"],
    "quic": ["quic", "http/3"],
    "dns": ["dns", "domain name system"],
    "mdns": ["mdns", "multicast dns", "bonjour", "avahi"],
    "http2": ["http/2", "http2", "hpack"],
    "http3": ["http/3", "http3", "qpack"],
    "tls13": ["tls 1.3", "tls 1.3 handshake"],
    "tls12": ["tls 1.2", "tls 1.2 handshake"],
    "bgp": ["bgp", "border gateway protocol"],
    "ospf": ["ospf", "open shortest path first"],
    "rsa": ["rsa"],
    "ecc": ["elliptic curve cryptography", "ecc", "ecdsa"],
    "password hashing": ["password hashing", "argon2", "pbkdf2", "blake2"],
    "waf": ["web application firewall", "waf"],
    "api gateway": ["api gateway", "token bucket", "kong"],
    "aes": ["aes", "advanced encryption standard"],
    "chacha": ["chacha20", "poly1305"],
    "oauth": ["oauth", "pkce", "proof key for code exchange"],
    "jwt": ["jwt", "json web token"],
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

    # Give a strong preference to passages containing
    # the exact entity asked about.

    if query_entities:

        for i, passage in enumerate(passages):

            passage_entities = find_entities(passage)

            matching = set(query_entities) & set(passage_entities)

            if matching:
                scores[i] += 0.22 * (
                    len(matching) / len(query_entities)
                )

    ranked = np.argsort(-scores)

    # For comparison questions, make sure that passages
    # for all requested entities can be considered.

    if query_entities:

        selected = []

        for entity in query_entities:

            for i in ranked:

                if entity in find_entities(passages[i]):
                    if int(i) not in selected:
                        selected.append(int(i))
                    break

        for i in ranked:

            if int(i) not in selected:
                selected.append(int(i))

        ranked = np.array(selected)

    return ranked[:max(top_k, len(query_entities))], scores, query_word, query_char, query_entities


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

    evidence = []
    used_sources = []

    for passage_index in ranked_indices:

        passage = passages[int(passage_index)]
        passage_entities = find_entities(passage)

        matching_entities = (
            set(query_entities) & set(passage_entities)
        )

        # When entities are specified, do not use an
        # unrelated passage as evidence.

        if query_entities and not matching_entities:
            continue

        candidates = []

        for i, source_index in enumerate(sentence_sources):

            if source_index != passage_index:
                continue

            sentence = sentences[i]

            if INJECTION_PATTERN.search(sentence):
                continue

            sentence_entities = find_entities(sentence)

            entity_match = (
                bool(set(query_entities) & set(sentence_entities))
                if query_entities else False
            )

            sentence_words = set(
                re.findall(r"[a-z0-9]+", normalize_text(sentence))
            )

            overlap = len(question_words & sentence_words)

            score = float(sentence_scores[i])

            # Prefer sentences that mention the requested entity.

            if entity_match:
                score += 0.30

            # Prefer sentences that contain specific query terms.

            score += min(overlap * 0.015, 0.12)

            candidates.append((score, sentence, entity_match))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []

        for score, sentence, entity_match in candidates:

            if score < 0.08:
                continue

            # Avoid unrelated sentences in entity-based questions.
            if query_entities and not entity_match:
                continue

            selected.append(sentence)

            if len(selected) >= 2:
                break

        if selected:

            evidence.extend(selected)
            used_sources.append(int(passage_index))

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

    # Abstain on clearly unrelated questions.

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

    # Keep no more than requested number of unique sources.

    used_sources = used_sources[:request.top_k]

    sources = [
        i for i in used_sources
    ]

    scores = [
        round(float(np.clip(combined[i], 0, 1)), 4)
        for i in sources
    ]

    # Keep evidence only from returned sources.

    allowed_sources = set(sources)

    final_evidence = []

    for sentence, source_index in zip(
        evidence,
        [
            source for source in used_sources
            for _ in range(2)
        ]
    ):

        if source_index in allowed_sources:
            final_evidence.append(sentence)

    answer = " ".join(final_evidence)

    if not answer:
        answer = " ".join(evidence)

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
