import re
from typing import List
from src.models import Paper, Chunk
from src.config import settings

def chunk_paper(paper: Paper, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> List[Chunk]:
    """
    Split a paper into chunks.
    For arXiv papers where we have title + abstract, we chunk the combined content.
    If the abstract fits in a single chunk (standard for abstracts ~150-250 words),
    it generates a single, high-fidelity chunk containing full paper context.
    """
    full_text = f"Title: {paper.title}\nAuthors: {', '.join(paper.authors)}\nYear: {paper.year}\nCategories: {', '.join(paper.categories)}\n\nAbstract: {paper.abstract}"
    
    words = full_text.split()
    if len(words) <= chunk_size:
        return [
            Chunk(
                chunk_id=f"{paper.arxiv_id}_c0",
                paper_id=paper.arxiv_id,
                text=full_text,
                index=0,
                metadata={
                    "title": paper.title,
                    "year": paper.year,
                    "categories": paper.categories,
                    "authors": paper.authors,
                    "arxiv_id": paper.arxiv_id
                }
            )
        ]

    chunks: List[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append(
            Chunk(
                chunk_id=f"{paper.arxiv_id}_c{idx}",
                paper_id=paper.arxiv_id,
                text=chunk_text,
                index=idx,
                metadata={
                    "title": paper.title,
                    "year": paper.year,
                    "categories": paper.categories,
                    "authors": paper.authors,
                    "arxiv_id": paper.arxiv_id
                }
            )
        )
        idx += 1
        start += (chunk_size - overlap)

    return chunks

def chunk_papers(papers: List[Paper]) -> List[Chunk]:
    """Process multiple papers into a flat list of chunks."""
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper))
    return all_chunks
