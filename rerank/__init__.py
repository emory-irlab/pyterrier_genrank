import pyterrier as pt

from rerank.api_keys import get_openai_api_key, get_azure_openai_args
from rerank.data import Candidate, Request, Query
from rerank.rank_gpt import SafeOpenai
from rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rerank.rankllm import PromptMode
from rerank.reranker import Reranker


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
                 use_azure_openai=False,
                 ushaped_positioning = False):
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
        # elif model_path.startswith('castorini/rankllama-') or model_path.startswith('rankllama-'):
        #     return PointwiseReranker(model_path)
        else:
            self.agent = RankListwiseOSLLM(model=model_path,
                                           num_few_shot_examples=num_few_shot_examples,
                                           num_gpus=num_gpus,
                                           prompt_mode=prompt_mode,
                                           context_size=context_size,
                                           variable_passages=variable_passages,
                                           system_message=system_message,
                                           prefix_instruction_fn=prefix_instruction_fn,
                                           suffix_instruction_fn=suffix_instruction_fn
                                           )
        self.reranker = Reranker(self.agent)
        self.text_key = text_key  # to allow fields other than 'text' to be used for reranking
        self.ushaped_positioning = ushaped_positioning

    def transform(self, retrieved):
        retrieved = retrieved.copy()
        query = Query(text=retrieved.iloc[0].query, qid=retrieved.iloc[0].qid)
        candidates = []
        if self.ushaped_positioning:
            for i, row in enumerate(retrieved.itertuples(index=False, name='Candidate')):
                candidate = Candidate(docid=row.docno, score=row.score, doc={'text': getattr(row, self.text_key)})
                if i % 2 == 0:
                    candidates.insert(i // 2, candidate)  # Insert at the next available "even" index (start)
                else:
                    candidates.insert(-(i // 2 + 1), candidate)  # Insert at the next available "odd" index (end)
        else:
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


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import pyterrier as pt
import pandas as pd


class PointwiseReranker(pt.Transformer):
    def __init__(self, model_path='castorini/rankllama-v1-7b-lora-passage', batch_size=1, text_key='text', device=None,
                 max_length=512):
        """
        Parameters:
          model_path: Hugging Face model identifier for the RankLLama pointwise model in PEFT format.
                      This should be the path to the adapter model (or the directory containing the adapter).
          batch_size: Number of candidate passages to process in one forward pass.
          text_key: Name of the column containing the passage text.
          device: Torch device (defaults to "cuda" if available, else "cpu").
          max_length: Maximum token length for model inputs.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.text_key = text_key
        self.max_length = max_length
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # Load the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config = PeftConfig.from_pretrained(model_path)
        # Load the base model first.
        base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
        # Wrap the base model with the PEFT adapter.
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()
    def transform(self, retrieved):
        """
        Rerank candidates in a DataFrame using the pointwise RankLLama model.
        Assumptions:
          - The input DataFrame `retrieved` has a column 'query' (same for all rows).
          - Each candidate row has a unique document id in 'docno' and text in the column specified by `text_key`.
        Returns:
          A DataFrame with updated 'score' and 'rank' columns.
        """
        retrieved = retrieved.copy()
        # Optionally preserve the original score.
        if 'score' in retrieved.columns:
            retrieved.rename(columns={'score': 'score_orig'}, inplace=True)
        # Assume the same query for all candidates.
        query = retrieved.iloc[0].query
        candidate_texts = retrieved[self.text_key].tolist()
        docnos = retrieved['docno'].tolist()
        scores = []
        # Process candidates in batches.
        for i in range(0, len(candidate_texts), self.batch_size):
            batch_texts = candidate_texts[i: i + self.batch_size]
            # Construct input prompt for each candidate.
            prompts = [f"Query: {query}\nPassage: {text}" for text in batch_texts]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                # If the model outputs a single logit (regression style), squeeze to get a scalar;
                # Otherwise, assume two logits and take the softmax probability for the positive (relevant) class.
                if outputs.logits.size(1) == 1:
                    batch_scores = outputs.logits.squeeze(-1).tolist()
                else:
                    batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].tolist()
            scores.extend(batch_scores)
        # Assign new scores and compute ranks based on descending score.
        retrieved['score'] = scores
        retrieved.sort_values(by='score', ascending=False, inplace=True)
        retrieved['rank'] = range(len(retrieved))
        return retrieved

