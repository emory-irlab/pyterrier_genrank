# PyTerrier_GenRank

![](https://img.shields.io/badge/PRs-welcome-brightgreen)
<img src="https://img.shields.io/badge/Version-1.0-lightblue.svg" alt="Version">
![Python version](https://img.shields.io/badge/lang-python-important)
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg)

The [PyTerrier🐕](https://github.com/terrier-org/pyterrier) Plugin for listwise generative rerankers like [RankVicuna](https://arxiv.org/abs/2309.15088)
and [RankZephyr](https://arxiv.org/abs/2312.02724). A PyTerrier wrapper over the implementation available at [RankLLM](https://github.com/castorini/rank_llm).

### Installation

```bash
pip install --upgrade git+https://github.com/emory-irlab/pyterrier_genrank.git
```

### PyTerrier Pipelines

Since this implementation uses listwise reranking, it is used a bit differently than other rerankers. 

```python
import pyterrier as pt
from rerank import LLMReRanker

dataset = pt.get_dataset("irds:vaswani")
docno2doctext = {doc['docno']: doc['text'] for doc in dataset.get_corpus_iter()}

bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
llm_reranker = LLMReRanker("castorini/rank_vicuna_7b_v1")
llm_reranker_pipe = lambda df : llm_reranker.rerank_pyt(df, 100, docno2doctext)

genrank_pipeline = bm25 >> llm_reranker_pipe
```
The LLMReRanker function can take any 🤗HuggingFace model id. It has been tested using the following two reranking models for TREC-DL 2019:

| Model                         | nDCG@10  |
|-------------------------------|----------|
| BM25                          | .48      |
| BM25 + rank_vicuna_7b_v1      | .67      |
| BM25 + rank_zephyr_7b_v1_full | .71      |

### Credits
Kaustubh Dhole, [IRLab](https://ir.mathcs.emory.edu/), Emory University
