# REPLUG: 
This includes an original implementation of **REPLUG: Retrieval-Augmented Black-Box Language Models**

## QA
### Step1: Build datastore file
The first step is to save embeddings of corpus. For LM tasks, we didn't include title in generating embeddings. 
Download the Wikipedia files from:
```
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

Generate embeddings&index for the corpus
```
OUTPUT_DIR=/path/to/output/embeddings
python generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever \
    --output_dir $OUTPUT_DIR  \
    --passages psgs_w100.tsv \
    --shard_id 0 --num_shards 1 \
```

### Step2: QA eval
Please also add the API key in the qa_final.py file.

```
MODEL=code-davinci-002 
DATASET=nq
QA_DATA=/path/to/qa/downloaded/nq_or_tqa/data

python -u downstream_eval/qa_final.py \
--engine $MODEL \
--data_dir $QA_DATA \
--task $DATASET \
--prompt_method open-book \
--save_prob \
--maxlen 10 \
--do_retrieval 1 \
--re_model_name_or_path facebook/contriever \
--passages psgs_w100.tsv \
--save_or_load_index \
--passages_embeddings "$OUTPUT_DIR/*" 
```


## MMLU
### Step1: Build datastore file
The first step is to save embeddings of corpus. For LM tasks, we didn't include title in generating embeddings. 
Download the Wikipedia files from:
```
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

Generate embeddings&index for the corpus
```
OUTPUT_DIR=/path/to/output/embeddings
python generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever \
    --output_dir $OUTPUT_DIR  \
    --passages psgs_w100.tsv \
    --shard_id 0 --num_shards 1 \
```

### Step2: MMLU eval

Download MMLU data from https://github.com/hendrycks/test

Please also add the API key in the mmlu_final.py file.
```
MODEL=code-davinci-002 
DATASET=mmlu
MMLU_DATA=/path/to/mmlu/downloaded/data

python -u downstream_eval/mmlu_final.py \
--engine $MODEL \
--data_dir $MMLU_DATA \
--task $DATASET \
--prompt_method open-book \
--save_prob \
--split test \
--maxlen 2 \
--do_retrieval 1 \
--re_model_name_or_path facebook/contriever \
--passages psgs_w100.tsv \
--save_or_load_index \
--passages_embeddings "$OUTPUT_DIR/*" 
```


## LM
### Step1: Build datastore file
The first step is to save embeddings of corpus. For LM tasks, we didn't include title in generating embeddings. 

```
ENCODE_PATH="./data/text.jsonl" # It could also be ENCODE_PATH=wikitext-2-v1

python3 generate_passage_embeddings.py \
        --model_name_or_path "facebook/contriever" \
        --passages $ENCODE_PATH \
        --output_dir data/embeddings \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
        --passage_maxlength 128 \
        --no_title  \
        --chunk_size 64 \
        --passage_maxlength 128 
```

### Step2: Save the ensemble probabilities 
First create a model config in preset_configs. One example model config:
```
  "model_type": "gpt2", # Use "gpt2" for all models that can be downloaded from Huggingface. 
  "model_name": "bigscience/bloom-1b7", # specify the model_name you want to use
  "context_len": 128, # any number is fine as it will be overwritten later
  "max_seq_len": 896, # any number is fine as it will be overwritten later
  "device": "cuda:0" 
 ```

Second, save log probabilites of your data using the following commands
```
ENCODE_PATH="./data/text.jsonl"
EMB_PATH="./embeddings/passages_00"
RETRIEVER="facebook/contriever"
MODEL_CONFIG="preset_configs/bloom-7b1.json"
ENSEMBLE_DOCS=10
python save_logprob_data.py       
       --model_config_path $MODEL_CONFIG  \
       --passages   $ENCODE_PATH  # the path to the raw corpus from step 1 \
       --passages_embeddings  $EMB_PATH # the path to encoded corpus from step 1 \
       --re_model_name_or_path $RETRIEVER  \
       --data   wikitext-2-v1    # dataset you want to use. Change the dataloading in line82/92 in save_logprob_data.py \
       --retrieved_max_length 128      \ # max length of each retrieved documents.
       --context_len 128     \ # Prior context used as the retrieval query
       --pred_len 768        \ # length of the next sentence following the prior context. This next sentence will be used to compute the log probability
       --output_path  outputs/ppl.data  \
       --ensemble $ENSEMBLE_DOCS    \ 
       --n_docs $ENSEMBLE_DOCS    \
       --save_or_load_index
```


## LSR finetuning:
```
python LSR_finetune/replug_lsr.py       
       --model_config_path $MODEL_CONFIG  \
       --passages   $ENCODE_PATH  # the path to the raw corpus from step 1 \
       --passages_embeddings  $EMB_PATH # the path to encoded corpus from step 1 \
       --re_model_name_or_path $RETRIEVER  \
       --data   wikitext-2-v1    # dataset you want to use. Change the dataloading in line82/92 in save_logprob_data.py \
       --retrieved_max_length 128      \ # max length of each retrieved documents.
       --context_len 128     \ # Prior context used as the retrieval query
       --pred_len 768        \ # length of the next sentence following the prior context. This next sentence will be used to compute the log probability
       --output_path  outputs/ppl.data  \
       --ensemble $ENSEMBLE_DOCS    \ 
       --n_docs $ENSEMBLE_DOCS    \
       --save_or_load_index
```



