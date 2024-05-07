import os

os.environ['HF_HOME'] = '/local/scratch/kdhole/multi-turn-tool-llm/huggingface'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GenerativeReranker(object):
    def __init__(self, model_id):
        with open('.hf_token', 'r') as file:
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
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def get_prompt(self, top_k_docs):
        # Construct the prompt for the ranker
        prompt = f"You are RankLLM, an intelligent assistant...\n"
        query = top_k_docs.iloc[0].query
        messages = [
            {"role": "system", "content": f'{prompt}'},
            {"role": "user", "content": f"Search Query: {query}.\n"}, ]
        for i, row in enumerate(top_k_docs.itertuples()):
            doc_text = docno2doctext.get(row.docno)
            messages.append({"role": "user", "content": f"[{i + 1}] {doc_text}\n"})
        return messages

    def rerank(self, r1, topk=100):
        r1 = r1.copy()
        messages = self.get_prompt(r1.head(topk))  # Assume top-100 for example
        ranked_output = prompt_model(messages)
        rank_map = {int(k.strip('[]')): i + 1 for i, k in enumerate(ranked_output.split(' > '))}
        # Apply rank and score updates
        r1.rename(columns={'score': 'score_0'}, inplace=True)
        r1['rank_new'] = r1['docno'].apply(lambda x: rank_map.get(x))
        r1['score'] = 1 / r1['rank_new']
        # Create r2 dataframe
        r2 = r1[['qid', 'docid', 'docno', 'rank_new', 'score_prev', 'score']]
        return r2
