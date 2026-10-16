import argparse
import re
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
    ChunkedSemanticSearch,

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

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type =str)
    chunk_parser.add_argument("--chunk_size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type = int)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max_chunk_size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)

    semantic_chunk_parser = subparsers.add_parser("embed_chunks")

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
        case "chunk":
            words = list(args.text.split())
            chunk_size = args.chunk_size
            if args.overlap>0:
                chunks = [words[i:i+chunk_size] for i in range(0,len(words),chunk_size-args.overlap)]
            else:
                chunks = [words[i:i+chunk_size] for i in range(0,len(words),chunk_size)]
            print(f"Chunking {chunk_size} words")
            for i,chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "semantic_chunk":
            ch = ChunkedSemanticSearch()
            chunks = ch.semantic_chunk(text=args.text, chunk_size=args.max_chunk_size, overlap=args.overlap)
            for i,chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "embed_chunks":
            ch = ChunkedSemanticSearch()
            movies = load_movies()
            chunk_embeddings = ch.load_or_create_chunk_embeddings(documents=movies)
            print(f"Generated {len(chunk_embeddings)} chunked embeddings.")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
