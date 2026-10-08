import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,

)
from lib.search_utils import (
    load_movies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify", help="Verify models")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for given text")
    embed_parser.add_argument("text", type=str, help = "Text")

    verify_embds_parser = subparsers.add_parser("verify_embeddings", help="bruh")

    query_embed_parser = subparsers.add_parser("embed_query", help="embed query")
    query_embed_parser.add_argument("query", type=str,help="query")

    semantic_search_parser = subparsers.add_parser("Semantic_Search", help="Semantic Search")
    semantic_search_parser.add_argument("query", type=str,help="query")
    semantic_search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(text=args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "Semantic_Search":
            semantic = SemanticSearch()
            movies = load_movies()
            embeddings = semantic.load_or_create_embeddings(movies)
            semantic.search(query=args.query,limit=args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
