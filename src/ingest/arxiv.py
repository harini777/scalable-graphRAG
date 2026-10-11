import os
import re
import time
import json
import logging
from typing import List, Dict, Optional, Generator
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

from src.config import settings
from src.models import Paper

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

def clean_text(text: str) -> str:
    """Clean abstract text: remove excess whitespace and LaTeX artifacts."""
    if not text:
        return ""
    # Replace newlines and excessive spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()

def parse_arxiv_entry(entry: ET.Element) -> Optional[Paper]:
    """Parse an Atom entry XML node into a Paper model."""
    try:
        id_elem = entry.find("atom:id", ATOM_NS)
        if id_elem is None or not id_elem.text:
            return None
        
        # arXiv ID is usually the last part of http://arxiv.org/abs/2106.09685v1
        raw_id = id_elem.text.strip()
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
        
        title_elem = entry.find("atom:title", ATOM_NS)
        title = clean_text(title_elem.text) if title_elem is not None and title_elem.text else "Untitled"
        
        summary_elem = entry.find("atom:summary", ATOM_NS)
        abstract = clean_text(summary_elem.text) if summary_elem is not None and summary_elem.text else ""
        
        published_elem = entry.find("atom:published", ATOM_NS)
        year = 2024
        if published_elem is not None and published_elem.text:
            try:
                year = int(published_elem.text[:4])
            except ValueError:
                pass
        
        authors = []
        for author_elem in entry.findall("atom:author", ATOM_NS):
            name_elem = author_elem.find("atom:name", ATOM_NS)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
                
        categories = []
        for cat_elem in entry.findall("atom:category", ATOM_NS):
            term = cat_elem.attrib.get("term")
            if term:
                categories.append(term)
                
        return Paper(
            arxiv_id=arxiv_id,
            title=title,
            year=year,
            abstract=abstract,
            categories=categories,
            authors=authors,
            citations=[]
        )
    except Exception as e:
        logger.warning(f"Failed to parse entry: {e}")
        return None

def fetch_arxiv_papers(
    categories: List[str] = ["cs.CL", "cs.LG"],
    max_results: int = 5000,
    batch_size: int = 100,
    delay_seconds: float = 3.0,
    output_path: Optional[Path] = None
) -> List[Paper]:
    """
    Fetch papers from arXiv API for given categories with polite rate limiting.
    Supports checkpointing to disk.
    """
    if output_path is None:
        output_path = settings.DATA_DIR / f"papers_{max_results}.jsonl"

    papers: List[Paper] = []
    seen_ids = set()

    # Load existing if available
    if output_path.exists():
        logger.info(f"Loading existing papers from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = Paper.model_validate_json(line)
                    papers.append(p)
                    seen_ids.add(p.arxiv_id)
        logger.info(f"Loaded {len(papers)} papers from existing cache.")

    if len(papers) >= max_results:
        return papers[:max_results]

    cat_query = "+OR+".join([f"cat:{c}" for c in categories])
    query = f"search_query={cat_query}&sortBy=submittedDate&sortOrder=descending"

    pbar = tqdm(total=max_results, initial=len(papers), desc="Fetching arXiv papers")

    start_index = len(papers)
    with open(output_path, "a", encoding="utf-8") as out_f:
        while len(papers) < max_results:
            fetch_count = min(batch_size, max_results - len(papers))
            url = f"{ARXIV_API_BASE}?{query}&start={start_index}&max_results={fetch_count}"
            
            try:
                resp = requests.get(url, headers={"User-Agent": "GraphRAG-Research/1.0"}, timeout=30)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    entries = root.findall("atom:entry", ATOM_NS)
                    if not entries:
                        logger.info("No more entries returned by arXiv API.")
                        break
                    
                    added = 0
                    for entry in entries:
                        paper = parse_arxiv_entry(entry)
                        if paper and paper.arxiv_id not in seen_ids:
                            papers.append(paper)
                            seen_ids.add(paper.arxiv_id)
                            out_f.write(paper.model_dump_json() + "\n")
                            out_f.flush()
                            added += 1
                            pbar.update(1)
                            if len(papers) >= max_results:
                                break
                    start_index += fetch_count
                else:
                    logger.warning(f"arXiv API error: HTTP {resp.status_code}. Sleeping...")
                    time.sleep(delay_seconds * 2)
            except Exception as e:
                logger.error(f"Error fetching batch at start={start_index}: {e}")
                time.sleep(delay_seconds * 2)

            time.sleep(delay_seconds)

    pbar.close()
    return papers

def create_seed_papers_dataset(output_path: Optional[Path] = None, count: int = 50) -> List[Paper]:
    """
    Creates a rich curated seed dataset of seminal ML papers (LoRA, QLoRA, RoBERTa, etc.)
    with known inter-paper relations, citations, and benchmark datasets.
    Ideal for rapid testing, offline development, and deterministic evaluation.
    """
    if output_path is None:
        output_path = settings.DATA_DIR / "papers_seed.jsonl"

    seed_data = [
        {
            "arxiv_id": "2106.09685",
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "year": 2021,
            "abstract": "We propose Low-Rank Adaptation (LoRA), which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. LoRA shows competitive performance on GLUE and SuperGLUE benchmarks compared to full fine-tuning.",
            "categories": ["cs.CL", "cs.LG"],
            "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
            "citations": ["1905.11946", "1706.03762"]
        },
        {
            "arxiv_id": "2305.14314",
            "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
            "year": 2023,
            "abstract": "We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). We evaluate QLoRA on the Vicuna benchmark and MMLU.",
            "categories": ["cs.LG", "cs.CL"],
            "authors": ["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
            "citations": ["2106.09685", "2302.13971"]
        },
        {
            "arxiv_id": "2309.15228",
            "title": "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models",
            "year": 2023,
            "abstract": "We present LongLoRA, an efficient fine-tuning approach that extends the context sizes of pre-trained large language models. Existing context extension methods require substantial computation. LongLoRA combines shifted sparse attention with LoRA to enable context window extension up to 100k tokens. We evaluate LongLoRA on long-context benchmarks including PG19 and Proof-Pile.",
            "categories": ["cs.CL", "cs.LG"],
            "authors": ["Yukang Chen", "Shengju Qian", "Haotian Tang", "Xin Lai", "Zhijian Liu", "Song Han", "Jiaya Jia"],
            "citations": ["2106.09685", "2302.13971"]
        },
        {
            "arxiv_id": "2302.13971",
            "title": "LLaMA: Open and Efficient Foundation Language Models",
            "year": 2023,
            "abstract": "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens using public datasets exclusively. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with state-of-the-art models like Chinchilla and PaLM on MMLU and GSM8K.",
            "categories": ["cs.CL", "cs.LG"],
            "authors": ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "Xavier Martinet", "Marie-Anne Lachaux", "Timothee Lacroix", "Baptiste Roziere", "Naman Goyal", "Eric Hambro", "Faisal Azhar", "Aurelien Rodriguez", "Armand Joulin", "Edouard Grave", "Guillaume Lample"],
            "citations": ["2005.14165"]
        },
        {
            "arxiv_id": "2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "year": 2020,
            "abstract": "We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance. We train GPT-3, an autoregressive language model with 175 billion parameters. Evaluated on SuperGLUE, TriviaQA, and LAMBADA, GPT-3 achieves strong performance under few-shot evaluation without task-specific fine-tuning.",
            "categories": ["cs.CL"],
            "authors": ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry", "Amanda Askell"],
            "citations": ["1905.11946"]
        },
        {
            "arxiv_id": "2205.14135",
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
            "year": 2022,
            "abstract": "Transformers are slow and memory-hungry on long sequences. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU HBM and SRAM. FlashAttention trains Transformers faster than existing attention baselines on the Long Range Arena (LRA) benchmark and language modeling tasks.",
            "categories": ["cs.LG"],
            "authors": ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Re"],
            "citations": ["1706.03762"]
        },
        {
            "arxiv_id": "2307.08691",
            "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
            "year": 2023,
            "abstract": "We propose FlashAttention-2, an improvement upon FlashAttention with better work partitioning and parallelism across thread blocks. FlashAttention-2 yields a 2x speedup compared to FlashAttention on A100 GPUs and scales to longer context lengths up to 64k tokens.",
            "categories": ["cs.LG"],
            "authors": ["Tri Dao"],
            "citations": ["2205.14135"]
        },
        {
            "arxiv_id": "1905.11946",
            "title": "Parameter-Efficient Transfer Learning for NLP",
            "year": 2019,
            "abstract": "Fine-tuning large pre-trained models is an effective transfer learning strategy. We propose adapter-based tuning for NLP, inserting compact adapter modules between layers of a frozen model. We evaluate adapter tuning on the GLUE benchmark, demonstrating near parity with full fine-tuning with only 3% parameter overhead.",
            "categories": ["cs.CL"],
            "authors": ["Neil Houlsby", "Andrei Giurgiu", "Stanislaw Jastrzebski", "Bruna Morrone", "Quentin de Laroussilhe", "Andrea Gesmundo", "Mona Attariyan", "Sylvain Gelly"],
            "citations": ["1810.04805"]
        },
        {
            "arxiv_id": "1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "year": 2018,
            "abstract": "We introduce BERT, a new language representation model designed to pre-train deep bidirectional representations from unlabeled text. BERT obtains state-of-the-art results on eleven natural language processing tasks, including GLUE benchmark, MultiNLI, and SQuAD.",
            "categories": ["cs.CL"],
            "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
            "citations": ["1706.03762"]
        },
        {
            "arxiv_id": "1907.11692",
            "title": "RoBERTa: A Robustly Optimized BERT Approach",
            "year": 2019,
            "abstract": "We present a replication study of BERT pre-training that carefully measures the impact of hyperparameter tuning and training set size. We find that BERT was significantly undertrained, and propose RoBERTa, which matches or exceeds post-BERT methods on GLUE, RACE, and SQuAD benchmarks.",
            "categories": ["cs.CL"],
            "authors": ["Yinhan Liu", "Myle Ott", "Naman Goyal", "Jingfei Du", "Mandar Joshi", "Danqi Chen", "Omer Levy", "Mike Lewis", "Luke Zettlemoyer", "Veselin Stoyanov"],
            "citations": ["1810.04805"]
        }
    ]

    papers = [Paper(**item) for item in seed_data]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(p.model_dump_json() + "\n")

    logger.info(f"Generated {len(papers)} seed papers at {output_path}")
    return papers
