import os
import sys
sys.path.append("../REPLUG")
import csv
from scipy.special import softmax
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
from time import sleep
from collections import defaultdict
import random
from retriever import Retriever
import openai
from tqdm import tqdm
from utils import *
import operator
from transformers import AutoTokenizer
from ipdb import set_trace as bp
import pandas as pd
from argument import add_lm_args, add_retriever_args

import random
random.seed(2022)

class KeyGen:
    def __init__(self) -> None:
        self.key_ind = 0
        self.api_keys = ["put your api keys"] 

    def get_key(self):
        self.key_ind += 1
        if self.key_ind >= len(self.api_keys):
            self.key_ind = 0
        return self.api_keys[self.key_ind]

key_generator = KeyGen()

def call_api(args, prompt, temp):
    tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    prompt = tokenizer.decode(tokenized[-7000 : ])

    ## A Single Prompt Step
    response = None
    while response is None:
        try:
            openai.api_key = key_generator.get_key()
            response = openai.Completion.create(
                engine=args.engine,
                prompt=prompt,
                max_tokens=args.maxlen,
                logprobs=4,
                temperature=temp,
                stream=False,
                stop="\n"
            )
            
        except:
            sleep(1)
            continue
   
    # print (response)
    if args.task == "mmlu":
        top_probs = []
        try:
            top_log_probs = response['choices'][0]["logprobs"]["top_logprobs"][0]
        except:
            print("!!!!!")
        for t in range(len(response['choices'][0]["logprobs"]["tokens"])):
            if response['choices'][0]["logprobs"]["tokens"][t] == "\n":
                break
            top_probs.append(response['choices'][0]["logprobs"]["token_logprobs"][t])
    else:
        top_probs = []
        top_tokens = []
        for t in range(len(response['choices'][0]["logprobs"]["tokens"])):
            if response['choices'][0]["logprobs"]["tokens"][t] == "\n":
                continue
            elif response['choices'][0]["logprobs"]["tokens"][t] == "<|endoftext|>":
                break
            top_probs.append(response['choices'][0]["logprobs"]["token_logprobs"][t])
            top_tokens.append(response['choices'][0]["logprobs"]["tokens"][t])
    perplexity = np.exp((np.mean(top_probs)))
    output = response['choices'][0]["text"].strip()

    if args.task == "mmlu":
        return output, prompt, (top_log_probs, perplexity)
    else:
        return output, prompt, (top_probs, perplexity)



def inference_one_ex(args, counter, prompt_batch, score_batch, eg):
    all_outputs = []
    all_probs = []
    for i, prompt in enumerate(prompt_batch):
        # bp()
        output, newprompt, probs = call_api(args, prompt, temp=0)
        ans = output
        
        ## exclude no-answer cases
        # if ans is not None:
        all_outputs.append(ans)
        # bp()
        all_probs.append(probs[1]*score_batch[i])
    
 
    ans2prob_list = defaultdict(list)
    for ans, prob in zip(all_outputs, all_probs):
        ans2prob_list[ans].append(prob)
    ans2prob = {k: sum(v) for k, v in ans2prob_list.items()}
    # bp()
    final_ans = max(ans2prob.items(), key=operator.itemgetter(1))[0]
    gold = eg["answer"]

    em = single_ans_em(final_ans, gold)
    return em


        
def retrieve_ex(demo, retriever):
    query = demo["question"]
    docs, scores = retriever.retrieve_passage([query])[0]
    # bp()
    plain_docs = [doc["text"] for doc in docs]
    return plain_docs, scores


def main():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    parser = argparse.ArgumentParser()
    parser = add_lm_args(parser)
    parser = add_retriever_args(parser)

    args = parser.parse_args()

    if args.do_retrieval:
        retriever = Retriever(args)
    else:
        retriever = None

    # load dataset
    all_counter = 0 
    all_em = 0

    
    '''
    data process
    '''
    with open(args.data_dir, "r") as f:
        data = json.load(f)
    test_set = data["testset"]
    demos = data["demos"][:16]


    print("test_set: ", len(test_set))
    # evaluate
    counter = 0
    demos_questions = [d["question"].strip() for d in demos]
    pbar = tqdm(test_set)

    # build prompt
    prompt_demo = ""
    prompt_demo_empty = ""
    if args.prompt_method in ["closed-book", "open-book"]:
        for demo in demos:
            # concat the top-1 doc
            if args.prompt_method == "open-book":
                docs, scores = retrieve_ex(demo, retriever)
                prompt_demo += f"Knowledge: {docs[0]}\n"
            prompt_demo += "Question: " + demo["question"] + "\n"
            # prompt_demo += "" + demo["question"] + "\n"
            answer = demo["answer"][0]
            prompt_demo += "Answer: " + answer.strip() + "\n\n"

            prompt_demo_empty += "Question: " + demo["question"] + "\n"
            prompt_demo_empty += "Answer: " + answer.strip() + "\n\n"

    # run over test example
    for eg in pbar:
        all_counter += 1
        # bp()
        prompt = prompt_demo
        prompt_empty = prompt_demo_empty
        if eg["question"].strip() in demos_questions:
            continue
        counter += 1
        
        if len(eg["question"].split()) > 400:
            eg["question"] = ' '.join(eg["question"].split()[-400 : ])
        
        prompt_batch = []
        score_batch = []
        if args.prompt_method == "open-book":
            docs, scores = retrieve_ex(eg, retriever)
            # contatenation version
            for doc, score in zip(docs, scores):
                prompt_cur = prompt
                prompt_cur += f"Knowledge: {doc}" + "\n"
                prompt_cur += "Question: " + eg["question"]  + "\n"
                prompt_cur += "Answer:"
                prompt_batch.append(prompt_cur)
                score_batch.append(score)

        elif args.prompt_method == "closed-book":
            prompt += "Question: " + eg["question"]  + "\n"
            prompt += "Answer:"
            prompt_batch.append(prompt)
            score_batch.append(1)

        score_batch = softmax(np.array(score_batch)).tolist()
        print("score_batch: ", score_batch)
        em = inference_one_ex(args, counter, prompt_batch, score_batch, eg)
        all_em += em

    if retriever is not None:
        retriever.dump_query2docs()

    print ("EM: {}/{}={}%".format(all_em, all_counter, (all_em / all_counter) * 100))


if __name__ == '__main__':
    main()