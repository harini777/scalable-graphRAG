import sys
from pathlib import Path
from src.config import settings
from src.index.vector_store import VectorIndex
from src.retrieve.baseline import PlainRAGPipeline

def main():
    print("=" * 60)
    print("  GraphRAG Research Assistant - Interactive Query CLI")
    print("=" * 60)
    
    index_path = settings.DATA_DIR / "indices" / "plain_rag"
    if not (index_path / "plain_rag.faiss").exists():
        print(f"Error: Vector index not found at {index_path}.")
        print("Please build it first with: python3 -m src.index.build_index")
        sys.exit(1)

    print("Loading vector index...")
    index = VectorIndex.load(index_path, prefix="plain_rag")
    rag = PlainRAGPipeline(vector_index=index, top_k=5)
    print(f"Index loaded successfully ({index.total_vectors()} vectors indexed).")
    print("\nAsk ANY question about ML research papers (or type 'exit' / 'quit' to quit):")
    print("-" * 60)

    while True:
        try:
            query = input("\nYour Question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break

            print("\nSearching papers and generating cited answer...")
            response = rag.query(query)

            print(f"\n--- Answer ({response.system_name} | Latency: {response.latency_ms:.1f}ms) ---")
            print(response.answer)

            print("\n--- Retrieved Sources & Citations ---")
            if response.citations:
                for c in response.citations:
                    print(f"  • [{c.paper_id}] {c.paper_title}")
            else:
                for r in response.retrieved_results:
                    print(f"  • [{r.paper_id}] {r.metadata.get('title', 'Unknown')}")

            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")

if __name__ == "__main__":
    main()
