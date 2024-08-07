# PyTerrier_GenRank

![](https://img.shields.io/badge/PRs-welcome-brightgreen)
<img src="https://img.shields.io/badge/Version-1.0-lightblue.svg" alt="Version">
![Python version](https://img.shields.io/badge/lang-python-important)
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg)

The [PyTerrier🐕](https://github.com/terrier-org/pyterrier) Plugin for listwise generative rerankers
like [RankGPT](https://aclanthology.org/2023.emnlp-main.923/), [RankVicuna](https://arxiv.org/abs/2309.15088), [RankZephyr](https://arxiv.org/abs/2312.02724). A PyTerrier wrapper over the implementation available
at [RankLLM](https://github.com/castorini/rank_llm).

### Installation

```bash
pip install --upgrade git+https://github.com/emory-irlab/pyterrier_genrank.git
```

### Example Usage

Since this implementation uses listwise reranking, it is used a bit differently than other rerankers.

```python
import pyterrier as pt
if not pt.started():
    pt.init()

from rerank import LLMReRanker

dataset = pt.get_dataset("irds:vaswani")

bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
llm_reranker = LLMReRanker("castorini/rank_vicuna_7b_v1")

genrank_pipeline = bm25 % 100 >> pt.text.get_text(index, 'text') >> llm_reranker

genrank_pipeline.search('best places to have Indian food')
```

If you want to use RankGPT, ensure that you have your [api key set in an environment file](rerank/api_keys.py). Then load the reranker with the OpenAI model string.
```python
llm_reranker = LLMReRanker("gpt-35-turbo-1106", use_azure_openai=True)
```

The LLMReRanker function can take any 🤗HuggingFace model id. It has been tested using the following two reranking models
for TREC-DL 2019:

| Model                         | nDCG@10 |
|-------------------------------|---------|
| BM25                          | .48     |
| BM25 + rank_vicuna_7b_v1      | .67     |
| BM25 + rank_zephyr_7b_v1_full | .71     |
| BM25 + gpt-35-turbo-1106      | .66     |
| BM25 + gpt-4-turbo-0409       | .71     |



The [reranker interface](rerank/__init__.py) takes additional parameters that could be modified.

```python
llm_reranker = LLMReRanker(model_path="castorini/rank_vicuna_7b_v1", 
                           num_few_shot_examples=0,
                           top_k_candidates=100,
                           window_size=20,
                           shuffle_candidates=False,
                           print_prompts_responses=False, step_size=10, variable_passages=True,
                           system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
                           prompt_mode: PromptMode = PromptMode.RANK_GPT,
                           context_size: int = 4096,
                           num_gpus = 1,
                           text_key = 'text',
                           use_azure_openai = False)
```

### Reference
```bibtex
@software{Dhole_PyTerrier_Genrank,
    author = {Dhole, Kaustubh},
    license = {Apache-2.0},
    institution = {Emory University},
    title = {{PyTerrier\_Genrank: The PyTerrier Plugin for generative rerankers}},
    url = {https://github.com/emory-irlab/pyterrier_genrank},
    year = {2024}
}
```
