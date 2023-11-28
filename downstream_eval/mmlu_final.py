import os
import sys
sys.path.append("../")
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
random.seed(0)

class KeyGen:
    def __init__(self) -> None:
        self.key_ind = 0
        if 'OPENAI_API_KEY' in os.environ:
            self.api_keys = os.environ['OPENAI_API_KEY'].split(',')
        else:
            print("OPENAI_API_KEY not found in environment variables. Calling OpenAI APIs will fail."
                  "OPENAI_API_KEY should be a comma-separated list of API keys."
                  "It can be set in .bashrc like: export OPENAI_API_KEY=key1,key2,key3"
            )
            self.api_keys = []

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
                stop="\n\n"
            )
        except:
            sleep(1)
            continue
    
    # print (response)
    if args.task == "mmlu":
        top_probs = []
        top_log_probs = response['choices'][0]["logprobs"]["top_logprobs"][0]
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
        return output, prompt, (top_log_probs, perplexity), response
    else:
        return output, prompt, (top_probs, perplexity), response

def inference_one_ex(args, counter, prompt_batch, score_batch, eg, return_predictions=False):
    all_outputs = []
    all_weighted_probs = []
    all_predictions = []
    for i, prompt in enumerate(prompt_batch):
        output, newprompt, probs, response = call_api(args, prompt, temp=0)
        ans = output
        all_outputs.append(ans)
        all_weighted_probs.append(probs[1]*score_batch[i])
        if return_predictions:
            all_predictions.append({
                "emsemble_id": i,
                "ans": ans,
                "prompt": prompt,
                "top_log_probs": probs[0],
                "prob": probs[1],
                "re_score": score_batch[i],
                "response": response
            })
    
    ans2prob_list = defaultdict(list)
    for ans, prob in zip(all_outputs, all_weighted_probs):
        ans2prob_list[ans].append(prob)
    ans2prob = {k: sum(v) for k, v in ans2prob_list.items()}

    final_ans = max(ans2prob.items(), key=operator.itemgetter(1))[0]
    gold = eg["answer"]
    em = single_ans_em(final_ans, gold)

    prediction_log = {
        "example_id": counter,
        "predicted_ans": final_ans,
        "gold": gold,
        "em": em,
        "esb_predictions": all_predictions   
    } if return_predictions else None

    return em, prediction_log

def data_from_csv_to_list(dev_df):
    demos = []
    for i in range(len(dev_df)):
        # print(dev_df.iloc[i, 0])
        one_d = {}
        one_d["question"] = f"{dev_df.iloc[i, 0]}\n(A) {dev_df.iloc[i, 1]}\n(B) {dev_df.iloc[i, 2]}\n(C) {dev_df.iloc[i, 3]}\n(D) {dev_df.iloc[i, 4]}"
        one_d["answer"] = dev_df.iloc[i, 5]
        demos.append(one_d)
    return demos
        
def retrieve_ex(demo, retriever):
    query = demo["question"]
    docs, scores = retriever.retrieve_passage([query])[0]
    plain_docs = [doc["text"] for doc in docs]
    return plain_docs, scores


def main():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    parser = argparse.ArgumentParser()
    parser = add_lm_args(parser)
    parser = add_retriever_args(parser)
    parser.add_argument("--save-predictions", default=False, action="store_true",
                        help="If set, save detailed prediction on disk.")
    parser.add_argument("--result-dir", type=str, default=None, 
                        help="Directory to save detailed predictions.")

    args = parser.parse_args()
    if args.save_predictions:
        assert(args.result_dir is not None)

    if args.do_retrieval:
        retriever = Retriever(args)
    else:
        retriever = None

    # load dataset
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    all_cors = []
    all_counter = 0 
    all_em = 0

    
    for i, subject in tqdm(enumerate(subjects)):

        cors = []        
        subject_em = 0
        subject_predictions = [] if args.save_predictions else None

        '''
        data process
        '''
        print(f"subject: {subject}")
        train_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.shots]
        val_df = pd.read_csv(os.path.join(args.data_dir, "val", subject + "_val.csv"), header=None)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        # build demos
        demos = data_from_csv_to_list(train_df) 
        
        # build test examples
        if args.split == "test":
            test_set = data_from_csv_to_list(test_df)
        elif args.split == "val":
            test_set = data_from_csv_to_list(val_df)
        print("test_set: ", len(test_set))
        # evaluate
        counter = 0
        demos_questions = [d["question"].strip() for d in demos]
        data_list = []
        pbar = tqdm(test_set)
        # bp()

        # build prompt
        prompt_demo = ""
        if args.prompt_method in ["closed-book", "open-book"]:
            for demo in demos:
                # concat the top-1 doc
                if args.prompt_method == "open-book":
                    docs, scores = retrieve_ex(demo, retriever)
                    prompt_demo += f"Knowledge: {docs[0]}\n"
                prompt_demo += "Question: " + demo["question"] + "\n"
                answer = demo["answer"]
                prompt_demo += "Answer: " + answer.strip() + "\n\n"

        # run over test example
        for eg in pbar:
            all_counter += 1
            # bp()
            prompt = prompt_demo
            pbar.set_description(f"Processing test examples from {subject}")
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
            score_batch = softmax(score_batch).tolist()
            em, prediction_log = inference_one_ex(args, counter, prompt_batch, score_batch, eg, return_predictions=args.save_predictions)
            all_em += em
            subject_em += em 
            cors.append(em)

            if args.save_predictions:
                subject_predictions.append(prediction_log)

        '''
        evaluation
        '''
        cors = np.array(cors)
        acc = np.mean(cors)            
        print ("\n\n")
        all_cors.append(cors)

        if args.save_predictions:
            print(f"{subject}[{args.split}] acc: {acc:.2f}")
            subject_results = {
                "subject": subject,
                "split": args.split,
                "acc": acc,
                "predictions": subject_predictions
            }
            out_json = os.path.join(args.result_dir, f"{subject}_results.json")
            with open(out_json, 'w') as o_f:
                json.dump(subject_results, o_f, indent=4)
                print(f"{subject} {args.split} predictions saved to {out_json}")
                print()

    weighted_acc = np.mean(np.concatenate(all_cors))
    print ("EM: {}/{}={}%".format(all_em, all_counter, (all_em / all_counter) * 100))
    print("MMLU overall acc: {:.3f}".format(weighted_acc))
    if args.save_predictions:
        out_json = os.path.join(args.result_dir, "overall_results.json")
        overall_results = {
            "weighted_acc": weighted_acc
        }
        with open(out_json, 'w') as o_f:
            json.dump(overall_results, o_f, indent=4)


if __name__ == '__main__':
    main()