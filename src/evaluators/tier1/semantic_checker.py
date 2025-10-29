"""
Semantic similarity checking using embeddings
"""
from typing import List, Tuple
import numpy as np
from src.utils.config import config


class SemanticChecker:
    """Check semantic similarity between claims and evidence using embeddings.
    
    Uses sentence transformer models to compute semantic similarity between
    text segments. This allows detection of unsupported claims even when
    the wording differs from the transcript.
    
    Key use cases:
        - Verify note sentences are supported by transcript
    - Find best evidence spans for claims
    - Detect paraphrases vs hallucinations
    
    Attributes:
        model_name (str): Name of the sentence transformer model
    model: Loaded SentenceTransformer model (None if unavailable)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize semantic checker with specified embedding model.

        Args:
            model_name (str): HuggingFace model identifier for sentence transformer.
            Defaults to all-MiniLM-L6-v2 (fast, good quality)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Attempt to load sentence transformer model with fallback.

        Tries to load the specified model. If loading fails, continues without
        the model (will return random embeddings for testing).
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f" Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode text strings into dense vector embeddings.

        Args:
            texts (List[str]): List of text strings to encode

        Returns:
            np.ndarray: 2D array of embeddings, shape (len(texts), embedding_dim)

        Note:
            If model is unavailable, returns random embeddings for testing purposes.
        """
        if self.model is None:
            # Fallback: return random embeddings (for testing)
            return np.random.rand(len(texts), 384)

        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two text strings.
    
        Encodes both texts and computes normalized cosine similarity,
        scaled to [0, 1] range where 1 means identical semantics and
        0 means completely unrelated.
        
        Args:
            text1 (str): First text
        text2 (str): Second text
        
        Returns:
            float: Similarity score in [0, 1], where:
            - 1.0 = semantically identical
        - 0.7+ = strong semantic overlap (typical paraphrase)
        - 0.4-0.7 = moderate similarity
        - <0.4 = weak/no similarity
        """
        if not text1 or not text2:
            return 0.0

        embeddings = self.encode([text1, text2])

        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Convert to [0, 1] range
        similarity = (similarity + 1) / 2

        return float(similarity)
    
    def find_best_support(self, claim: str, transcript_sentences: List[str], 
    top_k: int = 3) -> List[Tuple[str, float]]:
        """Find the most semantically similar transcript sentences for a claim.
    
        Computes similarity between the claim and all transcript sentences,
        returning the top-k best matches. Used for evidence retrieval.
        
        Args:
            claim (str): Claim or statement to find support for
        transcript_sentences (List[str]): List of candidate supporting sentences
        top_k (int): Number of top matches to return. Defaults to 3
        
        Returns:
            List[Tuple[str, float]]: Top-k matches as (sentence, similarity_score) tuples,
        sorted by similarity (highest first)
        
        Example:
        >>> checker = SemanticChecker()
        >>> claim = "Patient has diabetes"
        >>> transcript = ["Blood sugar elevated", "Takes metformin", "Weather is nice"]
        >>> checker.find_best_support(claim, transcript, top_k=2)
        [("Blood sugar elevated", 0.82), ("Takes metformin", 0.68)]
        """
        if not transcript_sentences:
            return []

        # Encode claim and all sentences
        claim_emb = self.encode([claim])[0]
        sent_embs = self.encode(transcript_sentences)

        # Compute similarities
        similarities = []
        for i, sent_emb in enumerate(sent_embs):
            dot_product = np.dot(claim_emb, sent_emb)
        norm1 = np.linalg.norm(claim_emb)
        norm2 = np.linalg.norm(sent_emb)

        if norm1 > 0 and norm2 > 0:
            sim = dot_product / (norm1 * norm2)
        sim = (sim + 1) / 2 # Convert to [0, 1]
        similarities.append((transcript_sentences[i], float(sim)))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def check_support(self, note_sentences: List[str], transcript_sentences: List[str],
    threshold: float = 0.72) -> List[Tuple[str, str, float]]:
        """Identify note sentences that lack sufficient support in the transcript.
    
    For each sentence in the note, finds the best matching transcript sentence.
    If the similarity is below the threshold, the sentence is flagged as
    potentially unsupported or hallucinated.
    
    Args:
        note_sentences (List[str]): Sentences from the generated note
    transcript_sentences (List[str]): Sentences from the original transcript
    threshold (float): Minimum similarity score for adequate support.
    Defaults to 0.72 (calibrated for medical notes)
    
    Returns:
        List[Tuple[str, str, float]]: List of unsupported claims, each containing:
        - note_sentence: The unsupported sentence from the note
    - best_match: Best matching transcript sentence (or "NO_EVIDENCE")
    - similarity: Similarity score between them
    
    Note:
        Filters out very short sentences (< min_sentence_length chars) to avoid
    false positives on headers, labels, etc.
    """
        unsupported = []

        min_length = config.get('tier1.min_sentence_length', 10)
        for note_sent in note_sentences:
            if len(note_sent.strip()) < min_length: # Skip very short sentences
                continue

        # Find best support
        supports = self.find_best_support(note_sent, transcript_sentences, top_k=1)

        if not supports:
            unsupported.append((note_sent, "NO_EVIDENCE", 0.0))
        elif supports[0][1] < threshold:
            unsupported.append((note_sent, supports[0][0], supports[0][1]))

        return unsupported

