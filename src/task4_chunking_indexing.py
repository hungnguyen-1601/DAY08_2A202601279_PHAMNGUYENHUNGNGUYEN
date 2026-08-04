"""
Task 4 - Chunking and indexing standardized markdown files into ChromaDB.

Pipeline:
    1. Read .md files from data/standardized/
    2. Split with RecursiveCharacterTextSplitter
    3. Embed chunks with SentenceTransformer all-MiniLM-L6-v2
    4. Upsert chunks and embeddings into persistent ChromaDB
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


class _FallbackRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            split_end = self._best_split_point(text, start, end)
            chunk = text[start:split_end].strip()
            if chunk:
                chunks.append(chunk)
            if split_end >= len(text):
                break
            start = max(split_end - self.chunk_overlap, start + 1)
        return chunks

    def _best_split_point(self, text: str, start: int, end: int) -> int:
        if end >= len(text):
            return len(text)
        window = text[start:end]
        for separator in self.separators:
            if not separator:
                continue
            index = window.rfind(separator)
            if index > 0:
                return start + index + len(separator)
        return end


def _get_recursive_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError:
        RecursiveCharacterTextSplitter = _FallbackRecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_documents() -> list[dict]:
    """
    Read all markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'path': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path).replace("\\", "/"),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents with RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}; each item is one chunk.
    """
    splitter = _get_recursive_splitter()
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict], model=None) -> list[dict]:
    """
    Embed chunks with sentence-transformers/all-MiniLM-L6-v2.

    The optional model parameter keeps unit tests fast and avoids loading the real model there.
    """
    if not chunks:
        return []
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    return chunks


def index_to_vectorstore(chunks: list[dict], collection=None):
    """
    Upsert chunks and embeddings into persistent ChromaDB.

    The optional collection parameter keeps unit tests independent from ChromaDB.
    """
    if not chunks:
        return None
    if collection is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    ids = [
        f"{chunk['metadata']['source']}_chunk_{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline():
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
