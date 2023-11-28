import argparse

parser = argparse.ArgumentParser()

def add_lm_args(parser):
    parser.add_argument('--apikey', type=str, required=False, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--task', type=str, help='specify the task that you want to evaluate')
    parser.add_argument("--data_dir", "-d", type=str, default="data")


    parser.add_argument('--prompt_method', type=str, default=None, help='specify the prompting method')
    parser.add_argument('--print', default=False, action='store_true', help='Whether to print out every prompt')
    parser.add_argument('--extract', default=False, action='store_true', help='Whether to add an additional answer extraction step')
    parser.add_argument('--subset', default=False, action='store_true', help='Whether to use a small subset for debugging')
    parser.add_argument('--subset_size', type=int, default=32, help='how many examples to sample for quick evaluation')
    parser.add_argument('--maxlen', type=int, default=256, help='max number of tokens to be generated')
    parser.add_argument('--shots', type=int, default=5, help='how many demos to use in the prompt')
    parser.add_argument('--no_unanswerable', default=False, action='store_true', help='Whether to filter out unanswerable questions in the demo')
    parser.add_argument('--label_shuffle', default=False, action='store_true', help='Whether to shuffle the gold labels')
    parser.add_argument('--save_prob', default=False, action='store_true', help='Whether to save top token logprobs and perplexity')
    parser.add_argument('--continue_from', type=int, default=0, help='evaluate on part of test set, starting from this index')
    parser.add_argument('--per_gpu_batch_size', type=int, default=64)
    return parser


def add_retriever_args(parser):
    # retrieval
    parser.add_argument('--do_retrieval', type=int, default=0,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--use-faiss-gpu', action="store_true", 
                        help='If enabled, use faiss GPU for retrieval inference')
    parser.add_argument('--num-gpus', type=int, default=-1)
    parser.add_argument('--ensemble', type=int, default=0,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--passages', type=str, default="wikitext-103-v1",
                        help='Path to passages (.tsv file)')  # wikitext-103-v1, wikitext-2-v1
    parser.add_argument('--passages_embeddings', type=str,
                        default="/private/home/swj0419/retro/baseline/contriever/wikitext_embeddings/passages_00", help='Glob path to encoded passages')
    parser.add_argument('--n_docs', type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--chunk_size', type=int, default=64,
                        help="Maximum number of words in a chunk")
    parser.add_argument('--normalize_text',
                        action='store_true', help="normalize text")
    parser.add_argument('--question_maxlength', type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument('--cache_dict', type=str, default="./query2docs.pk",
                        help='Path to passages (.tsv file)') 

    # 1024:
    parser.add_argument('--re_model_name_or_path', type=str, default="facebook/contriever",
                        help="path to directory containing model weights and config file")

    parser.add_argument('--projection_size', type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0,
                        help='Number of subquantizer used for vector quantization, if 0 flat index is used')
    parser.add_argument("--n_bits", type=int, default=8,
                        help='Number of bits per subquantizer')
    parser.add_argument('--indexing_batch_size', type=int, default=1000000,
                        help="Batch size of the number of passages indexed")
    parser.add_argument("--save_or_load_index", action='store_true',
                        help='If enabled, save index and load index if it exists')
    return parser