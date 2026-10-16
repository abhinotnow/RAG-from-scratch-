from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import os
import re
import json
from lib.search_utils import (
    CACHE_PATH,
    load_movies,
)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None 
        self.document_map = {}
        self.embeddings_path = CACHE_PATH/"movie_embeddings.npy"

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for document in documents:
            id = document["id"]
            self.document_map[id] = document
            movies.append(f"{document["title"]}: {document["description"]}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)

        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embeddings)
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents 
        for document in documents:
            id = document["id"]
            self.document_map[id] = document

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                self.build_embeddings(documents)
                return self.embeddings
        else:
            self.build_embeddings(documents)
            return self.embeddings

    def generate_embedding(self,text):
        if text.strip():
            embeddings = self.model.encode(list(text))
            return embeddings[0]
        else:
            raise ValueError("Please enter a proper string")

    def search(self, query, limit):
        if self.embeddings is not None:
            query_embed = self.generate_embedding(query)
            res = []
            for i in range(len(self.embeddings)):
                score = cosine_similarity(query_embed,self.embeddings[i])
                res.append((score,self.document_map[i+1]))
            res = sorted(res, key = lambda x: x[0], reverse=True)
            for result in res[:limit]:
                print(f"{result[1]["title"]} (score: {result[0]})")
                print(f"{result[1]["description"]}")
        else:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = CACHE_PATH/"chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_PATH/"chunk_metadata.json"

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        chunks = []
        chunk_metadata = []
        self.document_map.clear()
        for i,document in enumerate(documents):
            id = document["id"]
            self.document_map[id] = document

            if document["description"].strip():
                current_chunks = self.semantic_chunk(text=document["description"],chunk_size=4,overlap=1)
                num_chunks = len(current_chunks)
                chunks.extend(current_chunks)
                for j,chunk in enumerate(current_chunks):
                    temp = {}
                    temp["movie_idx"] = i
                    temp["chunk_idx"] = j
                    temp["total_chunks"] = num_chunks
                    chunk_metadata.append(temp)
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        with open(self.chunk_embeddings_path, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents 
        for document in documents:
            id = document["id"]
            self.document_map[id] = document

        if not os.path.exists(self.chunk_embeddings_path) or not os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = self.build_chunk_embeddings(documents=self.documents)
            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        else:
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

    def semantic_chunk(self, text, chunk_size, overlap):
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if overlap>0:
                chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0,len(sentences),chunk_size-overlap)]
            else:
                chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0,len(sentences),chunk_size)]
            return chunks

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is not None:
            query_embed = self.generate_embedding(query)
            res = []
            for i,chunk in enumerate(self.chunk_embeddings):
                entry = {}
                score = cosine_similarity(query_embed,chunk)
                metadata = self.chunk_metadata[i]
                entry["chunk_idx"] = metadata["chunk_idx"]
                entry["movie_idx"] = metadata["movie_idx"]
                entry["score"] = score
                res.append(entry)
            best = {}
            for entry in res:
                movie_idx = entry["movie_idx"]
                if movie_idx not in best:
                    best[movie_idx] = -1
                best[movie_idx] = max(best[movie_idx],entry["score"])
            best = dict(sorted(best.items(), key = lambda x: x[1], reverse=True))
            cntr = 0
            final = []
            for idx,score in best.items():
                if cntr == limit:
                    break
                doc = self.documents[idx]
                temp = {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 3),
                }
                final.append(temp)
                cntr+=1
            return final

        else:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

def verify_embeddings():
    semantic = SemanticSearch()
    movies = load_movies()
    embeddings = semantic.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_model():
        semantic = SemanticSearch()
        print(f"Model loaded: {semantic.model}")
        print(f"Max sequence length: {semantic.model.max_seq_length}")