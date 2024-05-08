# PyTerrier_Genrank
![](https://img.shields.io/badge/PRs-welcome-brightgreen) 
<img src="https://img.shields.io/badge/Version-1.0-lightblue.svg" alt="Version">
![Python version](https://img.shields.io/badge/lang-python-important)
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg) 
The PyTerrier Plugin for generative rerankers like [RankVicuna](https://arxiv.org/abs/2309.15088) and [RankZephyr](https://arxiv.org/abs/2312.02724).

### Installation
```bash
pip install --upgrade https://github.com/emory-irlab/pyterrier_genrank
```

### PyTerrier Pipelines
Since this implementation uses listwise reranking, it is used a bit differently than other rerankers.
You can pass the name of the model that you wish to run run against.

```python
import pyterrier as pt
from llm_reranker import LLMReRanker
dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")
genrank_pipeline = bm25 >> LLMReRanker("castorini/rank_vicuna_7b_v1")
```