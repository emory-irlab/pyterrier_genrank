import os
from typing import Any, Dict, List, Optional, Tuple, Union
os.environ['HF_HOME'] = '/local/scratch/kdhole/multi-turn-tool-llm/huggingface'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class GenerativeReranker(object):
    def __init__(self, model_id):
        with open('/local/scratch/kdhole/pyterrier_genrank/.hf_token', 'r') as file:
            token_access = file.read().strip()  # Read the token and remove any extra whitespace
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token_access)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token_access
        )
        self.device = self.model.device

    def prompt_model(self, messages):
        print(f'messages = {messages}')
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        # Check and truncate if necessary
        # if input_ids.shape[1] > self.max_length:
        #     input_ids = input_ids[:, :self.max_length]
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=200,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        ranking = self.tokenizer.decode(response, skip_special_tokens=True)
        print(f'ranking = {ranking}')
        return ranking

    def _get_prefix_for_rank_gpt_prompt(
            self, query: str, num: int
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

    def _get_suffix_for_rank_gpt_prompt(self, query: str, num: int) -> str:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

    def get_prompt(self, top_k_docs, docno2doctext):
        # Construct the prompt for the ranker
        num = len(top_k_docs)
        query = top_k_docs.iloc[0].query
        messages = self._get_prefix_for_rank_gpt_prompt(query, num)
        for i, row in enumerate(top_k_docs.itertuples()):
            doc_text = docno2doctext.get(row.docno)
            rank = i + 1
            messages.append({"role": "user", "content": f"[{rank}] {doc_text}\n"})
            messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
        messages.append(
            {
                "role": "user",
                "content": self._get_suffix_for_rank_gpt_prompt(query, num),
            }
        )
        return messages

    def rerank(self, r1, topk, docno2doctext):
        r1 = r1.copy()
        messages = self.get_prompt(r1.head(topk), docno2doctext)  # Assume top-100 for example
        ranked_output = self.prompt_model(messages)
        rank_map = {int(k.strip('[]')): i + 1 for i, k in enumerate(ranked_output.split(' > '))}
        # Apply rank and score updates
        max_rank = max(rank_map.values()) + 1
        r1.rename(columns={'score': 'score_0'}, inplace=True)
        r1['rank_new'] = r1['docno'].apply(lambda x: rank_map.get(x, max_rank))
        r1['score'] = 1 / r1['rank_new']
        # Create r2 dataframe
        r2 = r1[['qid', 'docid', 'docno', 'rank_new', 'score_0', 'score']]
        return r2