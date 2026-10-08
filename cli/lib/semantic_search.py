from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import os
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