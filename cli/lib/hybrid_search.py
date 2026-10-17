import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25search(query, limit)

    def normalize(self,results):
            if not results:
                return {}
            
            scores = [float(r["score"]) for r in results]
            minimum, maximum = min(scores), max(scores)

            if maximum == minimum:
                return {r["id"]: 1.0 for r in results}

            return {
                r["id"]: (float(r["score"]) - minimum) / (maximum - minimum)
                for r in results
            }

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        candidate_limit = limit * 500

        bm25_results = self._bm25_search(query, limit=candidate_limit)
        semantic_results = self.semantic_search.search_chunks(query, limit=candidate_limit)

        bm25_scores = self.normalize(bm25_results)
        semantic_scores = self.normalize(semantic_results)

        # Keep the union of candidates from both retrieval methods.
        all_ids = bm25_scores.keys() | semantic_scores.keys()

        results = []
        for movie_id in all_ids:
            bm25 = bm25_scores.get(movie_id, 0.0)
            semantic = semantic_scores.get(movie_id, 0.0)
            movie = self.semantic_search.document_map[movie_id]
            results.append({
                "id": movie_id,
                "title": movie["title"],
                "bm25_score": bm25,
                "semantic_score": semantic,
                "hybrid_score": alpha * bm25 + (1 - alpha) * semantic,
                "description": movie["description"],
            })

        return sorted(results, key=lambda r: r["hybrid_score"],reverse=True,)[:limit]

    def rank(self,results):
        return {
            result["id"]: i+1 for i,result in enumerate(results)
        }

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        if not k>=0 or not limit>=0:
            raise ValueError("Please input proper values for k and/or limit.")
        
        candidate_limit = limit * 500
        bm25_results = self._bm25_search(query, limit=candidate_limit)
        semantic_results = self.semantic_search.search_chunks(query, limit=candidate_limit)

        bm25_scores = self.rank(bm25_results)
        semantic_scores = self.rank(semantic_results)

        all_ids = bm25_scores.keys() | semantic_scores.keys()

        results = []
        for movie_id in all_ids:
            movie = self.semantic_search.document_map[movie_id]
            bm25_rank = bm25_scores.get(movie_id, 0.0)
            semantic_rank = semantic_scores.get(movie_id, 0.0)
            rrf_score = (
                    (1 / (k + bm25_rank) if movie_id in bm25_scores else 0)
                    + (1 / (k + semantic_rank) if movie_id in semantic_scores else 0)
                )
            results.append({
                "id": movie_id,
                "title": movie["title"],
                "description": movie["description"],
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "rrf_score": rrf_score
            })

        return sorted(results, key=lambda r: r["rrf_score"],reverse=True,)[:limit]


def score_norm(scores):
    if scores is not None:
        maxi = max(scores)
        mini = min(scores)
        res = [1.0]*len(scores) if maxi==mini else [(num-mini)/(maxi-mini) for num in scores]
        return res
    else:
        return []