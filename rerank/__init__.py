from rerank.api_keys import get_openai_api_key, get_azure_openai_args
from rerank.data import Candidate, Request, Query
from rerank.rank_gpt import SafeOpenai
from rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rerank.rankllm import PromptMode

from rerank.reranker import Reranker
import pandas as pd
import pyterrier as pt


class LLMReRanker(pt.Transformer):
    def __init__(self, model_path="castorini/rank_vicuna_7b_v1", num_few_shot_examples=0, top_k_candidates=100,
                 window_size=20,
                 shuffle_candidates=False,
                 print_prompts_responses=False, step_size=10, variable_passages=True,
                 system_message='You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.',
                 prefix_instruction_fn=lambda num,
                                              query: f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
                 suffix_instruction_fn=lambda num,
                                              query: f"Search Query: {query}. \nRank the {num} passages above. You should rank them based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.",
                 prompt_mode: PromptMode = PromptMode.RANK_GPT,
                 context_size: int = 4096,
                 num_gpus=1,
                 text_key='text',
                 use_azure_openai=False):
        self.window_size = window_size
        self.shuffle_candidates = shuffle_candidates
        self.top_k_candidates = top_k_candidates
        self.print_prompts_responses = print_prompts_responses
        self.step_size = step_size
        # Construct Rerank Agent
        if "gpt" in model_path or use_azure_openai:
            openai_keys = get_openai_api_key()
            self.agent = SafeOpenai(
                model=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                keys=openai_keys,
                system_message=system_message,
                prefix_instruction_fn=prefix_instruction_fn,
                suffix_instruction_fn=suffix_instruction_fn,
                **(get_azure_openai_args() if use_azure_openai else {}),
            )
        else:
            self.agent = RankListwiseOSLLM(model=model_path,
                                           num_few_shot_examples=num_few_shot_examples,
                                           num_gpus=num_gpus,
                                           prompt_mode=prompt_mode,
                                           context_size=context_size,
                                           variable_passages=variable_passages,
                                           system_message=system_message,
                                           )
        self.reranker = Reranker(self.agent)
        self.text_key = text_key  # to allow fields other than 'text' to be used for reranking

    def transform(self, retrieved):
        retrieved = retrieved.copy()
        query = Query(text=retrieved.iloc[0].query, qid=retrieved.iloc[0].qid)
        candidates = []
        for i, row in enumerate(retrieved.itertuples(index=False, name='Candidate')):
            candidate = Candidate(docid=row.docno, score=row.score, doc={'text': getattr(row, self.text_key)})
            candidates.append(candidate)
        request = Request(query=query, candidates=candidates)
        rerank_results = self.reranker.rerank(
            request,
            rank_end=self.top_k_candidates,
            window_size=min(self.window_size, self.top_k_candidates),
            shuffle_candidates=self.shuffle_candidates,
            logging=self.print_prompts_responses,
            step=self.step_size,
        )
        retrieved.rename(columns={'score': 'score_0'}, inplace=True)
        reranked_df = pd.DataFrame({
            'docno': [c.docid for c in rerank_results.candidates],
            'score': [1 / (r + 1) for r, c in enumerate(rerank_results.candidates)],  # reciprocal ranking
            'rank': [r for r, c in enumerate(rerank_results.candidates)]
        })
        result_df = retrieved.merge(reranked_df, on='docno', suffixes=('_orig', ''))
        return result_df
