# PyTerrier_GenRank

![](https://img.shields.io/badge/PRs-welcome-brightgreen)
<img src="https://img.shields.io/badge/Version-1.0-lightblue.svg" alt="Version">
![Python version](https://img.shields.io/badge/lang-python-important)
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg)

The PyTerrier 🐕Plugin for listwise generative rerankers like [RankVicuna](https://arxiv.org/abs/2309.15088)
and [RankZephyr](https://arxiv.org/abs/2312.02724). A PyTerrier wrapper over the implementation available at [RankLLM](https://github.com/castorini/rank_llm)

### Installation

```bash
pip install --upgrade https://github.com/emory-irlab/pyterrier_genrank
```

### PyTerrier Pipelines

Since this implementation uses listwise reranking, it is used a bit differently than other rerankers. 

```python
import pyterrier as pt
from llm_reranker import LLMReRanker

dataset = pt.get_dataset("irds:vaswani")
docno2doctext = {doc['docno']: doc['text'] for doc in dataset.get_corpus_iter()}
bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")
llm_reranker = LLMReRanker("castorini/rank_vicuna_7b_v1")
llm_reranker = lambda df : llm_reranker.rerank_pyt(df, 100, docno2doctext)

genrank_pipeline = bm25 >> llm_reranker
```
The LLMReRanker function can take any 🤗HuggingFace model id. Check [experiment.py](experiment.py) for running the complete pipeline over TREC-DL 2019 and evaluating the results.

### Citation
```bibtex
@software{Dhole_PyTerrier_Genrank_The_PyTerrier,
    author = {Dhole, Kaustubh},
    license = {Apache-2.0},
    title = {{PyTerrier\_Genrank: The PyTerrier Plugin for generative rerankers}},
    url = {https://github.com/emory-irlab/pyterrier_genrank}
}
```
While using, also cite the [RankLLM](https://github.com/castorini/rank_llm) repository.
