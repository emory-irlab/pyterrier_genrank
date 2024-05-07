import pyterrier as pt
from pyterrier.measures import * # don't uncomment this
import os
import pandas as pd
from global_utils import cleanup
from gen_reranker import GenerativeReranker
#print(f"Using GPUs : CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
import numpy as np
def print_seeds():
    print(f'numpy seed = {np.random.seed()}')

scratchpath = "/local/scratch/kdhole"
os.chdir(f'{scratchpath}/prompt-prf/terrier-prf')
os.environ['HF_HOME'] = '/local/scratch/kdhole/multi-turn-tool-llm/huggingface'
os.environ['IR_DATASETS_HOME']=f'{scratchpath}/prompt-prf/ir_datasets_download'
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

EVAL_METRICS = ['num_q', 'map', R@10, R@100, nDCG@10, nDCG@100, RR(rel=2), AP(rel=2)]

def get_index(EVALUATION_NAME, index_name, field=None):
    if index_name == "msmarco_passage":
        print(f"Loading Index {index_name}...")
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path)
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(pt_index_path)
        index = pt.IndexFactory.of(index_ref)
        print('Completed indexing')
        queries = dataset.get_topics()
        queries['query'] = queries['query'].apply(cleanup)
        return index, dataset, queries
    else:
        print(f"KD:No index selected of name {index_name}.")
        return None

def get_bm25_pipe(index_name, index):
    if index_name in ["trec-covid", "msmarco_passage", "msmarco_document"]:
        bm25 = pt.BatchRetrieve.from_dataset(index_name, 'terrier_stemmed', wmodel='BM25')
    else:
        bm25 = pt.BatchRetrieve(index, wmodel='BM25')
    return bm25

triplets = [['irds:msmarco-passage/trec-dl-2019/judged',  'msmarco_passage', 'text', 'text']]

def save_setting(result_dfs):
    combined_df = pd.concat(result_dfs, ignore_index=True)
    combined_df.to_csv(f'results.csv',index=False)

def get_docid2text(dataset, field='text'):
    docno2doctext = {doc['docno']: doc[field] for doc in dataset.get_corpus_iter()}
    return docno2doctext

def run_experiment(EVALUATION_NAME, index_name, field):
    index, dataset, queries = get_index(EVALUATION_NAME, index_name, field)
    bm25 = get_bm25_pipe(index_name,index)
    print(bm25.search('random query'))
    result_dfs = []
    result = pt.Experiment([bm25], queries, dataset.get_qrels(), EVAL_METRICS, ['BM25'], verbose=True, batch_size=10)
    result_dfs.append(result)
    save_setting(result_dfs)

def run_genrank_experiment(genranker, EVALUATION_NAME, index_name, field):
    index, dataset, queries = get_index(EVALUATION_NAME, index_name, field)
    bm25 = get_bm25_pipe(index_name,index)
    r1 = bm25.search('random query')
    reranked_frame = genranker.rerank(r1, 10)
    print(reranked_frame)

if __name__ == '__main__':
    triplet = triplets[0]

    EVALUATION_NAME = triplet[0];
    index_name = triplet[1];
    field = triplet[2];
    doc_field = triplet[3]

    run_experiment(EVALUATION_NAME, index_name, field)
    genranker = GenerativeReranker("meta-llama/Meta-Llama-3-8B-Instruct")
    run_genrank_experiment(genranker, EVALUATION_NAME, index_name, field)