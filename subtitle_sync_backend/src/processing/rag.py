"""
RAG utilities: compute embeddings and retrieve top-k relevant transcript segments
for each subtitle cue using cosine similarity.

We prefer sentence-transformers if available; otherwise, fall back to a deterministic
TF-IDF-based embedding to avoid heavy dependencies and external API keys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class TextVectorizer:
    """
    A simple wrapper that tries to use sentence-transformers if available, and
    otherwise falls back to a lightweight TF-IDF vectorizer on-the-fly.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _st_model: Any = None  # sentence-transformers model if available
    _use_st: bool = False

    # Fields for TF-IDF fallback
    _idf_vocab: Dict[str, float] = None
    _tokenizer: Any = None

    def __post_init__(self) -> None:
        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(self.model_name)
            self._use_st = True
        except Exception:
            # Fallback to TF-IDF: lazy-init; tokenizer is simple split
            self._st_model = None
            self._use_st = False
            self._idf_vocab = {}
            self._tokenizer = self._simple_tokenize

    def _simple_tokenize(self, text: str) -> List[str]:
        # Lowercase, split on whitespace, strip basic punctuation
        import re
        text = text.lower()
        # Replace non-alphanumeric with space
        text = re.sub(r"[^a-z0-9]+", " ", text)
        toks = [t for t in text.split() if t]
        return toks

    def _build_idf(self, corpus: List[str]) -> None:
        """
        Build a very small IDF dictionary for the corpus (fallback mode only).
        """
        import math
        doc_count = len(corpus) or 1
        df: Dict[str, int] = {}
        for doc in corpus:
            terms = set(self._tokenizer(doc))
            for t in terms:
                df[t] = df.get(t, 0) + 1
        self._idf_vocab = {}
        for term, cnt in df.items():
            # Use 1 + log(N / (1 + df))
            self._idf_vocab[term] = 1.0 + math.log(doc_count / (1.0 + cnt))

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        """
        Sparse TF-IDF vector (fallback) represented as term->weight.
        """
        tokens = self._tokenizer(text)
        if not tokens:
            return {}
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        # Normalize TF
        length = float(len(tokens))
        for t in list(tf.keys()):
            tf[t] = tf[t] / length

        # Multiply by IDF
        vec: Dict[str, float] = {}
        for t, tfv in tf.items():
            idf = self._idf_vocab.get(t, 1.0)  # unseen terms get idf 1.0
            vec[t] = tfv * idf
        return vec

    def _cosine_sparse(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        # dot
        dot = 0.0
        # Iterate smaller dict
        if len(a) < len(b):
            small, large = a, b
        else:
            small, large = b, a
        for k, v in small.items():
            bv = large.get(k)
            if bv is not None:
                dot += v * bv
        # norms
        def norm(x: Dict[str, float]) -> float:
            return math.sqrt(sum(v * v for v in x.values()))
        denom = norm(a) * norm(b)
        return (dot / denom) if denom > 0 else 0.0

    # PUBLIC_INTERFACE
    def encode(self, texts: List[str], corpus: Optional[List[str]] = None) -> Tuple[List[Any], bool]:
        """
        Encode a list of texts into dense vectors (if sentence-transformers) or sparse tf-idf dicts (fallback).

        Returns:
            (vectors, is_dense)
            vectors: list of numpy arrays (dense) OR list of dict[str,float] (sparse)
            is_dense: True if dense representation using sentence-transformers is used.
        """
        if self._use_st:
            # SentenceTransformers returns numpy arrays
            vecs = self._st_model.encode(texts, normalize_embeddings=True)
            return list(vecs), True
        else:
            # Build IDF on provided corpus if available; else use texts.
            base = corpus if corpus is not None else texts
            if not self._idf_vocab:
                self._build_idf(base)
            # Return sparse TF-IDF vectors
            return [self._tfidf_vector(t) for t in texts], False

    def cosine(self, a: Any, b: Any, is_dense: bool) -> float:
        """
        Compute cosine similarity for two vectors. Supports dense numpy arrays (if is_dense)
        or sparse dict vectors (fallback).
        """
        if is_dense:
            # a and b are numpy arrays
            import numpy as np  # local import to avoid global dependency complaints
            # They should already be normalized; still guard for zero vectors
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        else:
            # sparse dicts
            return self._cosine_sparse(a, b)


# PUBLIC_INTERFACE
def retrieve_top_k_for_cue(
    cue_text: str,
    entries: List[Dict[str, Any]],
    k: int = 10,
    vectorizer: Optional[TextVectorizer] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most semantically similar transcript segments from `entries`
    given a subtitle cue text. Uses cosine similarity over embeddings.

    Parameters:
        cue_text: The text of the subtitle cue.
        entries: List of transcript segments: each is {start: float, end: float, text: str}
        k: Number of results to return (default 10).
        vectorizer: Optional pre-initialized TextVectorizer to reuse models and IDF.

    Returns:
        A list of up to k entries augmented with a 'score' field, sorted by descending similarity.
    """
    if not entries:
        return []

    vec = vectorizer or TextVectorizer()

    # Build the corpus texts to support IDF in fallback and for potential ST batching
    corpus_texts = [str(e.get("text", "")) for e in entries]
    # Encode corpus and cue
    corpus_vecs, is_dense = vec.encode(corpus_texts, corpus=corpus_texts)
    cue_vecs, _ = vec.encode([cue_text], corpus=corpus_texts)
    cue_vec = cue_vecs[0]

    # Compute similarities
    scored: List[Tuple[int, float]] = []
    for i, ev in enumerate(corpus_vecs):
        s = vec.cosine(cue_vec, ev, is_dense=is_dense)
        scored.append((i, s))

    # Sort by score desc and take top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    top: List[Dict[str, Any]] = []
    for idx, score in scored[: max(1, k)]:
        item = dict(entries[idx])  # copy
        item["score"] = float(score)
        top.append(item)
    return top


# PUBLIC_INTERFACE
def retrieve_top_k_for_cues(
    cues: List[Dict[str, Any]],
    entries: List[Dict[str, Any]],
    k: int = 10,
) -> List[List[Dict[str, Any]]]:
    """
    Batch retrieval: for each cue (expects {'text': str, 'start': float, 'end': float}),
    return the top-k relevant transcript entries.

    Parameters:
        cues: list of subtitle cues (at least a 'text' field)
        entries: transcript entries
        k: top-k to return
    Returns:
        A list of lists: results[i] corresponds to cues[i], each inner list is sorted by desc similarity.
    """
    if not cues or not entries:
        return [[] for _ in cues]

    vec = TextVectorizer()
    corpus_texts = [str(e.get("text", "")) for e in entries]
    corpus_vecs, is_dense = vec.encode(corpus_texts, corpus=corpus_texts)

    # Precompute to avoid repeated work
    results: List[List[Dict[str, Any]]] = []
    for cue in cues:
        cue_text = str(cue.get("text", ""))
        cue_vecs, _ = vec.encode([cue_text], corpus=corpus_texts)
        cue_vec = cue_vecs[0]

        scored: List[Tuple[int, float]] = []
        for i, ev in enumerate(corpus_vecs):
            s = vec.cosine(cue_vec, ev, is_dense=is_dense)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)

        top: List[Dict[str, Any]] = []
        for idx, score in scored[: max(1, k)]:
            item = dict(entries[idx])
            item["score"] = float(score)
            top.append(item)
        results.append(top)
    return results
