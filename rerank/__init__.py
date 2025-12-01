import pyterrier as pt
import pyterrier_alpha as pta

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

    @pta.transform.by_query(add_ranks=False)
    def transform(self, retrieved):
        
        # validation and inspection support
        pta.validate.result_frame(retrieved, extra_columns=['query', self.text_key])
        if not len(retrieved):
            return pd.DataFrame([], columns=retrieved.columns.tolist() + ["score_0", "docno_orig"])
        
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

    @pta.transform.by_query(add_ranks=True)
    def transform(self, retrieved):
        """
        Rerank candidates in a DataFrame using the pointwise RankLLama model.
        Assumptions:
          - The input DataFrame `retrieved` has a column 'qid' that distinguishes the 'query' column.
          - Each candidate row has a unique document id in 'docno' and text in the column specified by `text_key`.
        Returns:
          A DataFrame with updated 'score' and 'rank' columns.
        """
        # validation and inspection support
        pta.validate.result_frame(retrieved, extra_columns=['query', self.text_key])
        if not len(retrieved):
            cols = retrieved.columns.tolist()
            if 'score' in cols:
                cols += ["score_orig"]
            return pd.DataFrame([], columns=cols)
        
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
        return retrieved


import torch
import math
import pyterrier as pt
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class Rank1Reranker(pt.Transformer):
    def __init__(
            self,
            model_name_or_path: str = "jhu-clsp/rank1-7b",
            batch_size: int = 999999999999,
            context_size: int = 16000,
            max_output_tokens: int = 8192,
            fp_options: str = "float16",
            num_gpus: int = 1,
            device: str = "cuda",
            force_rethink: int = 0,
            dataset_prompt: str = None,
            text_key: str = "text",
            **kwargs,
    ):
        """
        PyTerrier wrapper for the rank1 reasoning reranker.
        Parameters:
          model_name_or_path: Path or name of the rank1 model.
          batch_size: Maximum batch size (not used directly since vLLM handles batching).
          context_size: Maximum context length for the model.
          max_output_tokens: Maximum number of tokens to generate.
          fp_options: Floating point precision (e.g. 'float16').
          num_gpus: Number of GPUs to use for tensor parallelism.
          device: Device to run the model on.
          force_rethink: Number of times to force the model to rethink its answer.
          dataset_prompt: Optional prompt template; if provided, "FILL_QUERY_HERE" is replaced with the query.
          text_key: Name of the column in the input DataFrame containing the passage text.
        """
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.fp_options = fp_options
        self.num_gpus = num_gpus
        self.device = device
        self.force_rethink = force_rethink
        self.dataset_prompt = dataset_prompt
        self.text_key = text_key
        # Initialize the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Cache commonly used token IDs.
        self.true_token = self.tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer(" false", add_special_tokens=False).input_ids[0]
        self.think_token = self.tokenizer("<think>", add_special_tokens=False).input_ids[0]
        self.think_end_token = self.tokenizer("</think>", add_special_tokens=False).input_ids[-1]
        # Initialize the model via vLLM.
        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            gpu_memory_utilization=0.9,
            dtype=fp_options,
        )
        # Set up sampling parameters.
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_output_tokens,
            logprobs=20,
            stop=["</think> true", "</think> false"],
            skip_special_tokens=False
        )
    def return_prompt(self, query: str, doc_content: str, prompt: str) -> str:
        """
        Construct the prompt by inserting the query and passage.
        If a dataset_prompt is provided, replace "FILL_QUERY_HERE" with the query.
        """
        final_query = prompt.replace("FILL_QUERY_HERE", query) if prompt else query
        return (
            "Determine if the following passage is relevant to the query. "
            "Answer only with 'true' or 'false'.\n"
            f"Query: {final_query}\n"
            f"Passage: {doc_content}\n"
            "<think>"
        )
    def _fix_incomplete_responses(self, original_prompts, generated_texts):
        """
        Fix incomplete responses where the generated text is missing the closing </think> token.
        """
        cleaned_texts = []
        for text in generated_texts:
            text = text.rstrip()
            if not text.endswith(('.', '!', '?')):
                last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last_punct != -1:
                    text = text[:last_punct + 1]
            cleaned_texts.append(text.strip())
        forced_prompts = [
            f"{orig_prompt}\n{cleaned_text}\n</think>"
            for orig_prompt, cleaned_text in zip(original_prompts, cleaned_texts)
        ]
        new_sampling_args = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
            skip_special_tokens=False
        )
        outputs = self.model.generate(forced_prompts, new_sampling_args)
        all_final_texts = []
        all_token_counts = []
        all_scores = []
        for i in range(len(outputs)):
            try:
                text = outputs[i].outputs[0].text
                final_logits = outputs[i].outputs[0].logprobs[-1]
                assert self.false_token in final_logits and self.true_token in final_logits, \
                    f"final logits are missing true or false: {final_logits}"
            except Exception as e:
                print(f"Error: {e} on fixing error, setting score to 0.5")
                all_scores.append(0.5)
                all_token_counts.append(len(outputs[i].outputs[0].token_ids))
                all_final_texts.append(text)
                continue
            token_count = len(outputs[i].outputs[0].token_ids)
            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            all_final_texts.append(text)
            all_token_counts.append(token_count)
            all_scores.append(score)
        return all_final_texts, all_token_counts, all_scores
    def _prepare_prompts_for_rethink(self, prompts, texts, rethink_text: str = "Wait"):
        """
        Prepare revised prompts by appending a rethink instruction.
        """
        full_texts = [p + t for p, t in zip(prompts, texts)]
        stripped_texts = [t.split("</think>")[0] for t in full_texts]
        return [s + f"\n{rethink_text}" for s in stripped_texts], stripped_texts
    def _process_with_vllm(self, prompts):
        """
        Generate outputs for each prompt using vLLM.
        If the generated output is incomplete, attempt to fix it.
        """
        outputs = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_outputs = self.model.generate(batch_prompts, self.sampling_params)
            outputs.extend(batch_outputs)
        total_length = len(prompts)
        all_outputs = [None] * total_length
        all_output_token_counts = [None] * total_length
        all_scores = [None] * total_length
        incomplete_prompts = []
        incomplete_texts = []
        incomplete_indices = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            try:
                final_logits = output.outputs[0].logprobs[-1]
            except Exception as e:
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue
            if self.true_token not in final_logits or self.false_token not in final_logits:
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue
            token_count = len(output.outputs[0].token_ids)
            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            all_outputs[i] = text
            all_output_token_counts[i] = token_count
            all_scores[i] = score
        if incomplete_indices:
            fixed_texts, fixed_counts, fixed_scores = self._fix_incomplete_responses(
                incomplete_prompts, incomplete_texts
            )
            for orig_idx, (text, count, score) in zip(
                    incomplete_indices, zip(fixed_texts, fixed_counts, fixed_scores)
            ):
                all_outputs[orig_idx] = text
                all_output_token_counts[orig_idx] = count
                all_scores[orig_idx] = score
        return all_outputs, all_output_token_counts, all_scores
    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs) -> list:
        """
        Adapted from rank1.predict. Expects a list of tuples. Each tuple is either (query, passage)
        or (query, passage, instructions). Returns a list of scores.
        """
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs
        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() if q.strip() != i.strip() else q.strip() for i, q in
                       zip(instructions, queries)]
        if isinstance(passages[0], dict):
            passages = [f"{v['title']} {v['text']}" if 'title' in v else v['text'] for v in passages]
        prompts = [
            self.return_prompt(query, passage, self.dataset_prompt)
            for query, passage in zip(queries, passages)
        ]
        print(f"Example prompt: \n{prompts[0]}\n")
        texts, token_counts, scores = self._process_with_vllm(prompts)
        while self.force_rethink:
            revised_prompts, _ = self._prepare_prompts_for_rethink(prompts, texts)
            new_texts, new_token_counts, new_scores = self._process_with_vllm(revised_prompts)
            rethink_text = "Wait"
            texts = [prev + f"\n{rethink_text}" + new for prev, new in zip(texts, new_texts)]
            scores = new_scores
            token_counts = [prev + new for prev, new in zip(token_counts, new_token_counts)]
            self.force_rethink -= 1
        return scores
    
    @pta.transform.by_query(add_ranks=True)
    def transform(self, retrieved: pd.DataFrame) -> pd.DataFrame:
        """
        Rerank the candidates in the DataFrame using rank1.
        Assumptions:
          - Each row contains a unique document id ("docno") and the candidate passage text in column `self.text_key`.
        Returns:
          A DataFrame with new "score" and "rank" columns.
        """
        pta.validate.result_frame(retrieved, extra_columns=['query', self.text_key])
        if not len(retrieved):
            cols = retrieved.columns.tolist()
            return pd.DataFrame([], columns=cols)
        
        retrieved = retrieved.copy()
        # Extract the query from the first row.
        query = retrieved.iloc[0].query
        # Build input tuples for each candidate.
        input_to_rerank = [(query, row[self.text_key]) for _, row in retrieved.iterrows()]
        # Get scores via the predict method.
        scores = self.predict(input_to_rerank)
        # Assign scores and compute reciprocal ranking.
        retrieved['score'] = scores
        return retrieved
