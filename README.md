![](https://img.shields.io/badge/PRs-welcome-brightgreen)
<img src="https://img.shields.io/badge/Version-1.0-lightblue.svg" alt="Version">
![Python version](https://img.shields.io/badge/lang-python-important)
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg)

The [PyTerrier🐕](https://github.com/terrier-org/pyterrier) Plugin for listwise, pointwise and reasoning based (long CoT) generative rerankers
like [RankGPT](https://aclanthology.org/2023.emnlp-main.923/), [RankVicuna](https://arxiv.org/abs/2309.15088), [RankZephyr](https://arxiv.org/abs/2312.02724), [RankLLama](https://arxiv.org/abs/2310.08319). A PyTerrier wrapper over the implementation available
at [RankLLM](https://github.com/castorini/rank_llm), [Rank1](https://github.com/orionw/rank1). 

### Installation

```bash
pip install --upgrade git+https://github.com/emory-irlab/pyterrier_genrank.git
```

### Example Usage

Since this implementation uses listwise reranking, it is used a bit differently than other rerankers.

```python
import pyterrier as pt

from rerank import LLMReRanker

dataset = pt.get_dataset("irds:vaswani")

bm25 = pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed", wmodel="BM25")
llm_reranker = LLMReRanker("castorini/rank_vicuna_7b_v1")

genrank_pipeline = bm25 % 100 >> pt.text.get_text(dataset, 'text') >> llm_reranker

genrank_pipeline.search('best places to have Indian food')
```
The resulting pipeline looks as follows:
![Example pipeline diagram](pipeline.png)

If you want to use RankGPT, ensure that you have your [api key set in an environment file](rerank/api_keys.py). Then load the reranker with the OpenAI model string.
```python
llm_reranker = LLMReRanker("gpt-35-turbo-1106", use_azure_openai=True)
```

We recently added functionality for pointwise reranker RankLLama and reasoning based rerankers Rank1 too:
```python
from rerank import PointwiseReranker
llm_reranker = PointwiseReranker('castorini/rankllama-v1-7b-lora-passage')
```

```python
from rerank import Rank1Reranker
llm_reranker = Rank1Reranker("jhu-clsp/rank1-7b")
```



The LLMReRanker function can take any 🤗HuggingFace model id. It has been tested using the following two reranking models
for TREC-DL 2019:

| Model                                   | nDCG@10 |
|-----------------------------------------|---------|
| BM25                                    | .48     |
| BM25 + rank_vicuna_7b_v1                | .67     |
| BM25 + rank_zephyr_7b_v1_full           | .71     |
| BM25 + gpt-35-turbo-1106                | .66     |
| BM25 + gpt-4-turbo-0409                 | .71     |
| BM25 + gpt-4o-mini                      | .71     |
| BM25 + Llama-Spark (8B zero-shot)       | .61     |

Read the paper for detailed results [here](PyTerrier_GenRank_Paper.pdf). 

See [this notebook](examples/trecdl2019.ipynb) for an example of how to run experiments like this.

The [reranker interface](rerank/__init__.py) takes additional parameters that could be modified.

```python
llm_reranker = LLMReRanker(model_path="castorini/rank_vicuna_7b_v1", 
                           num_few_shot_examples=0,
                           top_k_candidates=100,
                           window_size=20,
                           shuffle_candidates=False,
                           print_prompts_responses=False, step_size=10, variable_passages=True,
                           system_message='You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.',
                           prefix_instruction_fn=lambda num, query: f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
                           suffix_instruction_fn=lambda num, query: f"Search Query: {query}. \nRank the {num} passages above. You should rank them based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.",
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
    title = {{PyTerrier-GenRank: The PyTerrier Plugin for Reranking with Large Language Models}},
    url = {https://github.com/emory-irlab/pyterrier_genrank},
    year = {2024}
}
```
