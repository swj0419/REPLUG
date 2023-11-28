
import numpy as np
import os
from tqdm import tqdm
from retriever import Retriever
from typing import Optional

import openai
import torch
import torch.nn as nn
import transformers
import utils

class LM:
    def get_perplexity_data(self, text) -> Optional[dict]:
        raise NotImplementedError

    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError

    def initialize_retriever(self, args):
        self.args = args
        if args.do_retrieval:
            self.retriever = Retriever(args)
        else:
            self.retriever = None

        
class GPT3LM(LM):

    def __init__(self, engine, context_len=1024, max_seq_len=2048, verbose=False, batch_size=16, optimizer=None, args=None):
        import openai
        self.engine = engine
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.wb = utils.WaitBlocker()
        self.verbose = verbose
        self.tmp = 1
        self.batch_size=batch_size
        self.optimzer=optimizer
        self.args = args

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    def forward_training(self, text):
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        

        batch_loss = []
        batch_index = 0
        # Remaining windows: input_tokens are context, pred_tokens are prediction
        for input_tokens, pred_tokens in tqdm(rolling_token_windows):
            retriever_loss = self.forward_training_single(input_tokens, pred_tokens)
            batch_loss.append(retriever_loss)
            if batch_index == self.batch_size:
                batch_loss = torch.stack(batch_loss)
                batch_loss = torch.mean(batch_loss)
                batch_loss.backward()
                batch_loss = []
                batch_index = 0
                self.optimizer.step()
                self.optimizer.zero_grad()



    def forward_training_single(self, input_tokens, pred_tokens):  
        query_id = input_tokens[:-len(pred_tokens)]
        # print("len(context):", len(query_id), "len(pred_tokens):", len(pred_tokens))
        query = self.tokenizer.decode(query_id)

        docs, scores = self.retriever.retrieve_passage([query])
        plain_docs = [doc["text"] for doc in docs]

        # encode the retrieved docs
        questions_embedding = self.embed_queries([query])
        passages_embedding = self.embed_queries(plain_docs)
        retriever_score = torch.einsum("id, ijd->ij", [questions_embedding, passages_embedding])
        all_gold_score = []
        for i in range(len(docs)):
            doc_str = plain_docs[i]
            doc_encodings = self.tokenizer.encode(doc_str)
            input_tokens_tmp = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
            block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens,)
            gold_score = block_output["logprobs"]
            all_gold_score.append(gold_score)
        all_gold_score = torch.FloatTensor(all_gold_score)
        retriever_loss = self.kldivloss(retriever_score, gold_score)
        return retriever_loss
          
    
    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.args.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.args.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)
    
    # noinspection DuplicatedCode
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        all_positions = []

        # Remaining windows: input_tokens are context, pred_tokens are prediction
        for input_tokens, pred_tokens in tqdm(rolling_token_windows):
            # ipdb.set_trace()
            # assert len(input_tokens) == 256
            # assert len(pred_tokens) == 512
            # bp()
            query_id = input_tokens[:-len(pred_tokens)]
            print("len(context):", len(query_id), "len(pred_tokens):", len(pred_tokens))
            # do retrieval
            if self.args.do_retrieval and (query_id != []):
                if self.args.random == 0:
                    query = self.tokenizer.decode(query_id)
                else:
                    query = "who is US president?"
                docs, scores = self.retriever.retrieve_passage([query])
                plain_docs = [doc["text"] for doc in docs]

                if self.args.ensemble == 0:
                    doc_str = "\n".join(plain_docs)
                    print(f"query: {[query]}\nretrieved doc: {[doc_str]}")
                    doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                    input_tokens = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
                    print("retrieve + context: ", len(input_tokens)-len(pred_tokens))
                else:
                    '''
                    a + b + c = log(e^log(a) + e^log(b) + e^log(c))
                    '''
                    logprobs_list = []
                    block_output = None
                    assert self.args.ensemble <= len(plain_docs)
                    
                    for i in range(self.args.ensemble):
                        doc_str = plain_docs[i]
                        doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                        input_tokens_tmp = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
                        block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens,)
                        logprobs_list.append(block_output["logprobs"])
                        # sum(np.isinf(block_output["logprobs"]))
                    # bp()
                    # block_output["logprobs"] = np.log(np.mean(np.exp(logprobs_list), axis=0))
                    block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list), dim=0) - np.log(len(logprobs_list))
                    block_output["logprobs"] = block_output["logprobs"].numpy()
            else:
                # bp()
                block_output = self.get_token_logprobs(input_tokens=input_tokens, pred_tokens=pred_tokens,)
            # bp()
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode('utf-8')),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens):
        pred_start = len(input_tokens) - len(pred_tokens) + 1
        # We're going to stitch together the input_tokens and pred_tokens
        # In the longest case, this gets us to length = max_seq_len+1 (which the API works with)
        assert input_tokens[pred_start:] == pred_tokens[:-1]
        token_ids = input_tokens + [pred_tokens[-1]]
        with self.wb.check_valid():
            response = openai.Completion.create(
                engine=self.engine,
                prompt=token_ids,
                max_tokens=0,
                temperature=0.0,
                logprobs=0,
                echo=True,
            )
        logprobs = np.array(response["choices"][0]["logprobs"]["token_logprobs"][pred_start:])
        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(token_ids))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(token_ids)[pred_start:])
            print("Perplexity:", np.exp(-logprobs.mean()))
            print()

        positions = np.arange(pred_start-1, pred_start-1 + len(token_ids[pred_start:]))

        return {
            "logprobs": logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)




