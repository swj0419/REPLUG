from retriever import Retriever
import numpy as np
import os
from typing import Optional
from tqdm import tqdm
import openai
import torch
import torch.nn as nn
import transformers
from ipdb import set_trace as bp
import utils as utils
import sys


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


class GPT2LM(LM):
    def __init__(self, model_name, device="cuda:0", context_len=512, max_seq_len=1024, verbose=False):
        self.model_name = model_name
        self.device = torch.device(device)
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        torch.set_grad_enabled(False)
        #  AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16).eval()
            # .to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name)
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

    # noinspection DuplicatedCode
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        # embed()
        # ipdb.set_trace()
        # bp()
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
            query_id = input_tokens[:-len(pred_tokens)]
            # do retrieval
            if self.args.do_retrieval and (query_id != []):
                query = self.tokenizer.decode(query_id)
                docs, scores = self.retriever.retrieve_passage([query])[0]
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
                    # block_output["logprobs"] = np.log(np.mean(np.exp(logprobs_list), axis=0))
                    # len(logprobs_list) = number of ensemble
                    # block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list), dim=0) - np.log(len(logprobs_list))
                    # apply softmax to scores 
                    scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
                    scores = torch.log(torch.FloatTensor(scores)).reshape(-1, 1)
                    scores = scores.repeat(1, len(logprobs_list[0]))
                    block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list)+scores, dim=0) 
                    block_output["logprobs"] = block_output["logprobs"].numpy()
            else:                # bp()
                block_output = self.get_token_logprobs(input_tokens=input_tokens, pred_tokens=pred_tokens,)
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
        input_tokens = torch.tensor(input_tokens).long().to(self.device)
        pred_tokens = torch.tensor(pred_tokens).long().to(self.device)
        input_tokens = input_tokens.unsqueeze(dim=0)
        pred_tokens = pred_tokens.unsqueeze(dim=0)
        # bp()
        output = self.model(input_tokens, return_dict=True)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        output.logits = output.logits.squeeze()
        pred_tokens = pred_tokens.squeeze()
        input_tokens = input_tokens.squeeze()
        neg_logprobs = loss_fct(
            output.logits[-len(pred_tokens):],
            pred_tokens,
        ).detach().cpu().numpy()
        # self.verbose=True
        if self.verbose:
            print("Context:", len(self.tokenizer.convert_ids_to_tokens(input_tokens)))
            print("Predicting:", len(self.tokenizer.convert_ids_to_tokens(pred_tokens)))
            print("Perplexity:", np.exp(neg_logprobs.mean()))
            print()

        positions = np.arange(len(input_tokens) -
                              len(pred_tokens), len(input_tokens))

        return {
            "logprobs": - neg_logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


def create_model(json_path):
    config = utils.read_json(json_path)
    model_type = config.pop("model_type")
    model = GPT2LM.create_from_config(config)
    return model


