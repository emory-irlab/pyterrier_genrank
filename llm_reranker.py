from data import Candidate, Request, Query
from rerank.rank_listwise_os_llm import RankListwiseOSLLM

from rerank.reranker import Reranker
import pandas as pd

class LLMReRanker(object):
    def __init__(self, model_path="castorini/rank_vicuna_7b_v1", top_k_candidates=100, window_size=20, shuffle_candidates=False,
                 print_prompts_responses=False, step_size=10):
        self.window_size = window_size
        self.shuffle_candidates = shuffle_candidates
        self.top_k_candidates = top_k_candidates
        self.print_prompts_responses = print_prompts_responses
        self.step_size = step_size
        self.agent = RankListwiseOSLLM(model=model_path,
                                       num_few_shot_examples=0,
                                       num_gpus=1,
                                       variable_passages=True,
                                       system_message='You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.',
                                       )
        self.reranker = Reranker(self.agent)
    def rerank_pyt(self, retrieved, topk, docno2doctext):
        retrieved = retrieved.copy().head(topk)
        query = Query(text=retrieved.iloc[0].query, qid=retrieved.iloc[0].qid)
        candidates = []
        for i, row in enumerate(retrieved.itertuples()):
            doc_text = docno2doctext.get(row.docno)
            if doc_text is not None:
                candidate = Candidate(docid=row.docno, score=row.score, doc={"text": doc_text})
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
        rank = 1
        for candidate in rerank_results.candidates:
            candidate.score = 1/rank
            rank +=1
        reranked_df = pd.DataFrame({
            'docno': [c.docid for c in rerank_results.candidates],
            'score' : [c.score for c in rerank_results.candidates],
        })
        result_df = retrieved.merge(reranked_df, on='docno', suffixes=('_orig', ''))
        return result_df

