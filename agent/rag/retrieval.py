"""
TF-IDF based retrieval system for document corpus
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Document:
    """Represents a document chunk with metadata"""
    def __init__(self, chunk_id: str, content: str, source: str, score: float = 0.0):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.score = score

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "score": self.score
        }


class TFIDFRetriever:
    """TF-IDF based document retriever"""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[Document] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.doc_vectors = None
        self._load_and_chunk_documents()

    def _load_and_chunk_documents(self):
        """Load all markdown files and split into chunks"""
        if not self.docs_dir.exists():
            raise ValueError(f"Documents directory not found: {self.docs_dir}")

        for filepath in self.docs_dir.glob("*.md"):
            self._process_file(filepath)

        if not self.chunks:
            raise ValueError("No documents loaded")

        # Build TF-IDF vectors
        chunk_texts = [chunk.content for chunk in self.chunks]
        self.doc_vectors = self.vectorizer.fit_transform(chunk_texts)

    def _process_file(self, filepath: Path):
        """Process a single markdown file into chunks"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        source = filepath.stem

        # Split by headers or paragraphs
        # First try to split by headers (##)
        sections = re.split(r'\n(?=##?\s)', content)

        for idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Create chunk ID
            chunk_id = f"{source}::chunk{idx}"

            # Store chunk
            chunk = Document(
                chunk_id=chunk_id,
                content=section,
                source=source
            )
            self.chunks.append(chunk)

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve top-k most relevant chunks for the query"""
        if not self.chunks:
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Compute similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Create result documents with scores
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            result = Document(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                score=score
            )
            results.append(result)

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Document:
        """Get a specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
