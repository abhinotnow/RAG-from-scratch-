import argparse
from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search,
    BM25_k1,
    BM25_B,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="search using BM25")

    search_parser = subparsers.add_parser("tf", help="Search movies using BM25")
    search_parser.add_argument("doc_id", type=int, help="Document ID to search")
    search_parser.add_argument("term", type=str, help="Term to search")

    search_parser = subparsers.add_parser("idf", help="Calculate IDF for a given term")
    search_parser.add_argument("term", type=str, help="Term to search")

    search_parser = subparsers.add_parser("tfidf", help="Calculate TFIDF for a given term")
    search_parser.add_argument("doc_id", type=int, help="Document ID to search")
    search_parser.add_argument("term", type=str, help="Term to search")


    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument( "k1", type=float, nargs="?", default=BM25_k1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    #bm25search_parser.add_argument("limit", type=int, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            res = search_command(args.query, 5)
            for i,title in enumerate(res):
                print(f"{i}. {title}")
                
        case "build":
            build_command()

        case "tf":
            id = tf_command(args.doc_id,args.term)
            print(id)

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_command(doc_id=args.doc_id,term=args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(doc_id = args.doc_id, term = args.term, k1 = args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            bm25scores = bm25_search(query=args.query)
            for i,score in enumerate(bm25scores):
                print(f"{i+1}. {score}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()