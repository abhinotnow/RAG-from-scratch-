import argparse
from lib.hybrid_search import (
    score_norm,
    HybridSearch,
)
from lib.search_utils import (
    load_movies,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparser.add_parser("normalize")
    norm_parser.add_argument("numbers", type=int, nargs="*")

    weighted_search_parser = subparser.add_parser("weighted_search")
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--limit", type=int, default=5)
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)

    rrf_search_parser = subparser.add_parser("rrf_search")
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument("--k", type=int ,default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            res = score_norm(args.numbers)
            if res is not None:
                for score in res:
                    print(f"* {score: .4f}")
        case "weighted_search":
            movies = load_movies()
            hybrid = HybridSearch(documents = movies)
            res = hybrid.weighted_search(query=args.query, alpha=args.alpha,limit=args.limit)
            for i,result in enumerate(res):
                print(f"{i+1}. {result["title"]} \n Hybrid Score: {result["hybrid_score"]:.4f} \n BM25: {result["bm25_score"]:.4f}, Semantic: {result["semantic_score"]} \n {result["description"][:100]}...")
        case "rrf_search":
            movies = load_movies()
            hybrid = HybridSearch(documents = movies)
            res = hybrid.rrf_search(query=args.query, k=args.k,limit=args.limit)
            for i,result in enumerate(res):
                print(f"{i+1}. {result["title"]} \n RRF Score: {result["rrf_score"]:.4f} \n BM25 Rank: {result["bm25_rank"]:.4f}, Semantic Rank: {result["semantic_rank"]} \n {result["description"][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()