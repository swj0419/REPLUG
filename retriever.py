import copy
import pickle
import numpy as np
import faiss
import glob
from logging import getLogger
import time
import os
from tqdm import tqdm
import torch
from datasets import load_dataset
import ipdb
import re
import sys
from ipdb import set_trace as bp

import index_utils.index
import index_utils.contriever
import index_utils.dragon
import index_utils.utils
import index_utils.slurm
import index_utils.data_contriever
from index_utils.evaluation import calculate_matches
import index_utils.normalize_text


logger = getLogger()


def ends_mid_sentence(passage):
    return not re.search(r'[.!?"]$', passage)


def get_incomplete_sentence(passage, return_from_the_end=False):
    if return_from_the_end:
        sentences = re.split(r'\s*[.!?]\s+', passage)
        return sentences[-1].rstrip()
    else:
        match = re.match(r'^.*?[.!?]', passage)
        if match:
            return match.group()
        else:
            print("Warning: cannot detect sentence boundary", passage)
            return passage


def remove_incomplete_sentences(passage: str, remove_from_beginning=True, remove_from_the_end=True) -> str:
    """
    Removes the incomplete sentences at the beginning and end of a passage.
    
    Args:
        passage (str): The passage to process.
        
    Returns:
        str: The passage with incomplete sentences removed.
    """
    if remove_from_beginning:
        # Remove any text before the first sentence-ending punctuation
        passage = re.sub(r'^.*?[.!?]\s', '', passage, flags=re.DOTALL)

    if remove_from_the_end:
        # Remove any text after the last sentence-ending punctuation
        passage = re.sub(r'([.!?])[^.!?]*$', r'\1', passage, flags=re.DOTALL)

    return passage


class Retriever():
    """
    Retriever class for retrieving data from the database.
    """

    def __init__(self, args):
        """
        Initialize the Retriever class.
        """
        self.args = args
        # ipdb.set_trace()
        if 'dragon' in args.re_model_name_or_path.lower():
            self.model, self.tokenizer = index_utils.dragon.load_retriever(
                args.re_model_name_or_path
            )
        else:
            self.model, self.tokenizer = index_utils.contriever.load_retriever(
                args.re_model_name_or_path
            )
        self.model.cuda()
        self.model.eval()
        # if not args.no_fp16:
        #     self.model = self.model.half()

        self.index = index_utils.index.Indexer(
            args.projection_size, args.n_subquantizers, args.n_bits)

        # index all passages
        input_paths = glob.glob(args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, 'index.faiss')
        # bp()
        if args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f'Indexing passages from files {input_paths}')
            start_time_indexing = time.time()
            self.index_encoded_data(
                self.index, input_paths, args.indexing_batch_size)
            print(f'Indexing time: {time.time() - start_time_indexing:.1f} s.')
            if args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        if args.use_faiss_gpu and faiss.get_num_gpus() > 0:
            start_time_converting = time.time()
            if args.num_gpus != -1:
                num_gpus = args.num_gpus
            else:
                num_gpus = faiss.get_num_gpus()
            print(f"Using {num_gpus} GPU devices found, converting to GPU index")
            cloner_options = faiss.GpuMultipleClonerOptions()
            cloner_options.shard = True
            cloner_options.useFloat16 = True
            self.index.index = faiss.index_cpu_to_all_gpus(self.index.index, co=cloner_options, ngpu=num_gpus)
            print(f'Conversion time: {time.time() - start_time_converting:.1f} s.')

        if os.path.exists(args.cache_dict):
            self.query2docs = pickle.load(open(args.cache_dict, "rb"))
        else:
            self.query2docs = {}

        # load passages
        if args.passages.startswith("wikitext"):
            passages = index_utils.data_contriever.process_huggingface_dataset(
                args.passages, args.chunk_size)
        else:
            passages = index_utils.data_contriever.load_passages(args.passages)

        self.passage_id_map = {x['id']: x for x in passages}

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f'Loading file {file_path}')
            with open(file_path, 'rb') as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack(
                (allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(
                    index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(
                index, allembeddings, allids, indexing_batch_size)

        print('Data indexing completed.')

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def embed_queries(self, queries):
        self.model.eval()
        embeddings, batch_question = [], []
        with torch.no_grad():

            for k, q in enumerate(queries):
                if self.args.normalize_text:
                    q = index_utils.normalize_text.normalize(q)
                batch_question.append(q)
                # print("batch_question: ", batch_question)
                if len(batch_question) == self.args.per_gpu_batch_size or k == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=self.args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda()
                                     for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        # print(f'Questions embeddings shape: {embeddings.size()}')
        return embeddings.numpy()

    def dump_query2docs(self):
        with open(self.args.cache_dict, "wb") as f:
            pickle.dump(self.query2docs, f)

    def retrieve_passage(self, queries):
        # print("queries: ", queries)
        # queries = [" person is laughing, type when the person is typing"]
        if len(queries) == 1 and queries[0] in self.query2docs:
            return [self.query2docs[queries[0]]]
        else:
            questions_embedding = self.embed_queries(queries)
            # get top k results
            start_time_retrieval = time.time()
            top_ids_and_scores = self.index.search_knn(
                questions_embedding, self.args.n_docs)
            print(f"Retrieval completed in {time.time() - start_time_retrieval}s")
            # retrieve passages
            # list: [[doc_ids, scores], ...]
            num_queries = len(top_ids_and_scores)
            assert(num_queries == len(queries))
            top_docs_and_scores = []
            for i in range(num_queries):
                docs = [] 
                for doc_id in top_ids_and_scores[i][0]:
                    doc = copy.deepcopy(self.passage_id_map[doc_id])
                    logger.debug("Before:", doc["text"])
                    if hasattr(self.args, "ra_truncate_broken_sents") and self.args.ra_truncate_broken_sents:
                        doc["text"] = remove_incomplete_sentences(doc["text"])
                    # TODO: The sentence rounding approach assumes passages are in consecutive order
                    elif hasattr(self.args, "ra_round_broken_sents") and self.args.ra_round_broken_sents:
                        if int(doc_id) > 0:
                            pre_doc_id = str(int(doc_id) - 1)
                            pre_doc = self.passage_id_map[pre_doc_id]
                            if ends_mid_sentence(pre_doc["text"]):
                                first_half = get_incomplete_sentence(pre_doc["text"], return_from_the_end=True)
                                doc["text"] = first_half + " " + doc["text"].lstrip()
                            if ends_mid_sentence(doc["text"]):
                                if int(doc_id) < len(self.passage_id_map) - 1:
                                    next_doc_id = str(int(doc_id) + 1)
                                    next_doc = self.passage_id_map[next_doc_id]
                                    second_half = get_incomplete_sentence(next_doc["text"], return_from_the_end=False)
                                    doc["text"] = doc["text"].rstrip() + " " + second_half
                    logger.debug("After:", doc["text"])
                    logger.debug()
                    docs.append(doc)

                scores = [score for score in top_ids_and_scores[i][1]]
                top_docs_and_scores.append((docs, scores))
                self.query2docs[queries[i]] = (docs, scores)
            return top_docs_and_scores
