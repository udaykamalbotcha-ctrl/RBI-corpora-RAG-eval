"""Vector Store and Local Embedding Pipeline for RBI Regulatory Corpora.

Integrates ChromaDB and Sentence-Transformers to create a local vector index
for regulatory text chunks. Supports:
- Dual client modes: Local PersistentClient (default) or Docker HttpClient (port 8000)
- Local open-source sentence-transformers embedding model (all-MiniLM-L6-v2)
- Strategy-tagged collections (fixed, recursive, clause_aware)
- Hierarchical metadata preservation (source_doc, chapter, section, clause_number, pages)
- Validation test utility for similarity search and clause tag retrieval
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.chunker import Chunk, create_chunker
from src.ingest import RegulatoryPDFParser, RegulatoryClause


class RegulatoryVectorStore:
    """Manages embedding generation and ChromaDB vector indexing."""

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_COLLECTION = "rbi_regulations"
    DEFAULT_PERSIST_DIR = "./chroma_db"

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL_NAME,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        host: Optional[str] = None,
        port: int = 8000,
        mode: str = "auto",  # 'auto', 'persistent', or 'http'
    ):
        self.collection_name = collection_name
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.host = host or os.environ.get("CHROMA_SERVER_HOST", "localhost")
        self.port = port or int(os.environ.get("CHROMA_SERVER_PORT", 8000))
        self.mode = mode.lower()

        # Initialize Embedding Model
        print(f"[VectorStore] Loading embedding model: '{self.model_name}'...")
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)
        dim = getattr(self.model, "get_embedding_dimension", None)
        if callable(dim):
            dim_val = dim()
        else:
            dim_val = self.model.get_sentence_embedding_dimension()
        print(f"[VectorStore] Embedding model loaded (dim={dim_val}).")

        # Initialize ChromaDB Client
        self.client = self._init_client()
        self.collection = self._init_collection()

    def _init_client(self) -> chromadb.ClientAPI:
        """Initialize either HttpClient or PersistentClient based on mode and availability."""
        if self.mode == "http":
            print(f"[VectorStore] Connecting to Chroma HTTP server at {self.host}:{self.port}...")
            return chromadb.HttpClient(host=self.host, port=self.port)
        elif self.mode == "persistent":
            print(f"[VectorStore] Using local persistent ChromaDB at '{self.persist_directory}'...")
            return chromadb.PersistentClient(path=self.persist_directory)
        else:  # auto mode
            # Try HTTP first if server is running, fallback to persistent
            try:
                print(f"[VectorStore] Checking for Chroma HTTP server at {self.host}:{self.port}...")
                http_client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(chroma_client_request_timeout=2)
                )
                http_client.heartbeat()
                print(f"[VectorStore] Connected to running Chroma HTTP container at {self.host}:{self.port}")
                return http_client
            except Exception:
                print(f"[VectorStore] Chroma HTTP server not detected. Using local PersistentClient at '{self.persist_directory}'")
                return chromadb.PersistentClient(path=self.persist_directory)

    def _init_collection(self):
        """Get or create the Chroma collection configured with cosine similarity."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "model": self.model_name},
        )

    def ingest_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Compute embeddings and ingest chunks with their metadata into ChromaDB."""
        if not chunks:
            print("[VectorStore] No chunks provided for ingestion.")
            return 0

        print(f"[VectorStore] Preparing to ingest {len(chunks)} chunks into collection '{self.collection_name}'...")

        total_ingested = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.to_chroma_metadata() for c in batch]

            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=len(batch),
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            # Upsert into ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_ingested += len(batch)
            print(f"  -> Ingested {total_ingested}/{total_chunks} chunks ({(total_ingested/total_chunks)*100:.1f}%)")

        print(f"[VectorStore] Ingestion complete. Total items in collection: {self.collection.count()}")
        return total_ingested

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        strategy_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query the vector store with a question and return ranked results with clause tags."""
        query_emb = self.model.encode([query_text], normalize_embeddings=True).tolist()

        query_where = {}
        if where:
            query_where.update(where)
        if strategy_filter:
            query_where["strategy"] = strategy_filter

        chroma_where = query_where if query_where else None

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where=chroma_where,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx]
            similarity = round(1.0 - distance, 4)
            hits.append({
                "chunk_id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "similarity_score": similarity,
                "distance": distance,
            })

        return hits

    def validate_retrieval(
        self,
        mock_question: str = "What are Covered Transactions under the CVA framework?",
        top_k: int = 3,
    ) -> bool:
        """Validation test function to ensure query retrieves chunks with correct clause tags."""
        print("\n" + "=" * 65)
        print("RUNNING RETRIEVAL VALIDATION TEST")
        print("=" * 65)
        print(f"Query: \"{mock_question}\"\n")

        hits = self.query(query_text=mock_question, top_k=top_k)

        if not hits:
            print("[FAIL] No chunks retrieved from vector store!")
            return False

        print(f"Retrieved {len(hits)} matching chunks:\n")
        all_tagged = True
        for i, hit in enumerate(hits):
            meta = hit["metadata"]
            clause_num = meta.get("clause_number", "MISSING")
            chapter = meta.get("chapter", "MISSING")
            src_doc = meta.get("source_doc", "MISSING")
            strat = meta.get("strategy", "MISSING")

            print(f"[{i+1}] Score: {hit['similarity_score']:.4f}")
            print(f"    Clause Tag : {clause_num}")
            print(f"    Chapter    : {chapter}")
            print(f"    Source Doc : {src_doc}")
            print(f"    Strategy   : {strat}")
            print(f"    Text Preview: {hit['text'][:180]}...")
            print("-" * 55)

            if clause_num == "MISSING" or chapter == "MISSING":
                all_tagged = False

        if all_tagged:
            print("[PASS] Validation succeeded: All retrieved chunks contain valid clause tags & hierarchy metadata.")
            return True
        else:
            print("[WARN] Some retrieved chunks are missing clause tags.")
            return False


def build_and_populate_vector_store(
    data_dir: str = "data",
    strategy: str = "clause_aware",
    persist_dir: str = "./chroma_db",
    validate: bool = True,
):
    """End-to-end pipeline: Parse PDFs -> Chunk -> Embed -> Index -> Validate."""
    print("=" * 65)
    print("RBI REGULATORY CORPORA RAG EVALUATION - PHASE 1 PIPELINE")
    print("=" * 65)

    # 1. Parse PDF documents
    parser = RegulatoryPDFParser()
    clauses = parser.parse_corpus(data_dir)
    if not clauses:
        print("[Error] No clauses parsed. Pipeline aborted.")
        return None

    # 2. Chunk text
    print(f"\n[Chunker] Applying '{strategy}' chunking strategy...")
    chunker = create_chunker(strategy)
    chunks = chunker.chunk_clauses(clauses)
    print(f"[Chunker] Generated {len(chunks)} chunks.")

    # 3. Embed & Ingest into ChromaDB
    collection_name = f"rbi_{strategy}"
    vstore = RegulatoryVectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    vstore.ingest_chunks(chunks)

    # 4. Run retrieval validation
    if validate:
        vstore.validate_retrieval()

    return vstore


if __name__ == "__main__":
    target_data = sys.argv[1] if len(sys.argv) > 1 else "data"
    strat = sys.argv[2] if len(sys.argv) > 2 else "clause_aware"
    build_and_populate_vector_store(data_dir=target_data, strategy=strat, validate=True)
