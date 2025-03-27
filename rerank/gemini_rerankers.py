import os
import pyterrier as pt
import pandas as pd

# Import necessary types and abstract base classes.
from google import genai
from google.genai import types

from rerank.api_keys import get_gemini_api_key
from rerank.rankllm import RankLLM, PromptMode
from rerank.data import Candidate, Request, Query
from rerank.reranker import Reranker

###############################################
# Google LLM class (Listwise re-ranking version)
###############################################

class SafeGoogle(RankLLM):
    def __init__(self,
                 model: str = "gemini-2.5-pro-exp-03-25",
                 context_size: int = 1024,
                 prompt_mode: PromptMode = PromptMode.RANK_GPT,
                 num_few_shot_examples: int = 0,
                 system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
                 prefix_instruction_fn=lambda num, query: f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the query: {query}.",
                 suffix_instruction_fn=lambda num, query: f"Search Query: {query}. Rank the {num} passages above. Provide the ranking in the format [1] > [2] > ... with the most relevant first.",
                 ):
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        self.system_message = system_message
        self.prefix_instruction_fn = prefix_instruction_fn
        self.suffix_instruction_fn = suffix_instruction_fn
        self.client = genai.Client(api_key=get_gemini_api_key())
        self.generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=150,  # adjust as needed
            response_mime_type="text/plain",
        )

    def run_llm(self, prompt, current_window_size: int = None) -> tuple:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        response_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=self.generate_content_config,
        ):
            response_text += chunk.text
        token_count = self.get_num_tokens(response_text)
        return response_text, token_count

    def create_prompt(self, result, rank_start: int, rank_end: int) -> tuple:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        # Build the prompt by concatenating the system message, prefix, candidate passages, and suffix.
        prompt = self.system_message + "\n"
        prompt += self.prefix_instruction_fn(num, query) + "\n"
        rank = 1
        for cand in result.candidates[rank_start:rank_end]:
            # Here we assume the candidate passage text is in the "text" field.
            content = cand.doc.get('text', '')
            prompt += f"[{rank}] {content}\n"
            rank += 1
        prompt += self.suffix_instruction_fn(num, query)
        token_count = self.get_num_tokens(prompt)
        return prompt, token_count

    def get_num_tokens(self, prompt) -> int:
        # A simple (but not exact) token count approximation using whitespace splitting.
        if isinstance(prompt, str):
            return len(prompt.split())
        elif isinstance(prompt, list):
            return sum(len(m.get("content", "").split()) for m in prompt)
        else:
            return 0

    def cost_per_1k_token(self, input_token: bool) -> float:
        # Adjust if you have cost estimates; here we return 0.
        return 0.0

    def num_output_tokens(self, current_window_size: int = None) -> int:
        # For simplicity, assume a fixed output token count (you can refine this estimate).
        return 50

##################################################
# Listwise (windowwise) re-ranker using Google LLM
##################################################

class LLMGoogleReRanker(pt.Transformer):
    def __init__(self,
                 model_path="gemini-2.5-pro-exp-03-25",
                 num_few_shot_examples=0,
                 top_k_candidates=100,
                 window_size=20,
                 shuffle_candidates=False,
                 print_prompts_responses=False,
                 step_size=10,
                 variable_passages=True,
                 system_message='You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.',
                 prefix_instruction_fn=lambda num, query: f"I will provide you with {num} passages, each indicated by a number identifier []. Rank the passages based on their relevance to query: {query}.",
                 suffix_instruction_fn=lambda num, query: f"Search Query: {query}. Rank the {num} passages above. Provide the ranking in the format [1] > [2] > ... with the most relevant first.",
                 prompt_mode: PromptMode = PromptMode.RANK_GPT,
                 context_size: int = 1024,
                 text_key='text',
                 ushaped_positioning=False):
        self.window_size = window_size
        self.shuffle_candidates = shuffle_candidates
        self.top_k_candidates = top_k_candidates
        self.print_prompts_responses = print_prompts_responses
        self.step_size = step_size
        # Always use Google in this transformer (adjust the condition as needed)
        self.agent = SafeGoogle(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            system_message=system_message,
            prefix_instruction_fn=prefix_instruction_fn,
            suffix_instruction_fn=suffix_instruction_fn,
        )
        self.reranker = Reranker(self.agent)
        self.text_key = text_key
        self.ushaped_positioning = ushaped_positioning

    def transform(self, retrieved):
        retrieved = retrieved.copy()
        query = Query(text=retrieved.iloc[0].query, qid=retrieved.iloc[0].qid)
        candidates = []
        if self.ushaped_positioning:
            for i, row in enumerate(retrieved.itertuples(index=False, name='Candidate')):
                candidate = Candidate(docid=row.docno, score=row.score, doc={'text': getattr(row, self.text_key)})
                if i % 2 == 0:
                    candidates.insert(i // 2, candidate)
                else:
                    candidates.insert(-(i // 2 + 1), candidate)
        else:
            for row in retrieved.itertuples(index=False, name='Candidate'):
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
            'score': [1 / (r + 1) for r, c in enumerate(rerank_results.candidates)],
            'rank': [r for r, c in enumerate(rerank_results.candidates)]
        })
        result_df = retrieved.merge(reranked_df, on='docno', suffixes=('_orig', ''))
        return result_df

##################################################
# Pointwise re-ranker using Google LLM
##################################################

class PointwiseGoogleReranker(pt.Transformer):
    def __init__(self, model_path="gemini-2.5-pro-exp-03-25", batch_size=1, text_key='text', max_length=512):
        self.model_path = model_path
        self.batch_size = batch_size
        self.text_key = text_key
        self.max_length = max_length
        self.client = genai.Client(api_key=get_gemini_api_key())
        self.generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=50,  # expect a short output (i.e. a numerical score)
            response_mime_type="text/plain",
        )

    def _call_google(self, prompt):
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        response_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model_path,
            contents=contents,
            config=self.generate_content_config,
        ):
            response_text += chunk.text
        return response_text

    def transform(self, retrieved):
        retrieved = retrieved.copy()
        if 'score' in retrieved.columns:
            retrieved.rename(columns={'score': 'score_orig'}, inplace=True)
        query = retrieved.iloc[0].query
        candidate_texts = retrieved[self.text_key].tolist()
        scores = []
        # Process each candidate individually.
        for text in candidate_texts:
            # Create a prompt asking for a relevance score (between 0 and 1)
            prompt = f"Query: {query}\nPassage: {text}\nPlease provide a relevance score between 0 and 1."
            response = self._call_google(prompt)
            try:
                score = float(response.strip())
            except Exception as e:
                score = 0.0
            scores.append(score)
        retrieved['score'] = scores
        retrieved.sort_values(by='score', ascending=False, inplace=True)
        retrieved['rank'] = range(len(retrieved))
        return retrieved

