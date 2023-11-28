from retriever import Retriever
from datasets import load_dataset
from tqdm import tqdm
from IPython import embed
import sys
sys.path.append("./")
import utils as utils
import models as models
import lm_dataformat
import argparse
import torch
from tqdm import auto as tqdm_lib
from ipdb import set_trace as bp
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--data', required=True,
                        default="wikitext-103", type=str)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--doc_indices_path', type=str, default=None)
    parser.add_argument('--per_gpu_batch_size', type=int, default=64)


    # retrieval
    parser.add_argument('--do_retrieval', type=int, default=0,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--use-faiss-gpu', action="store_true", 
                        help='If enabled, use faiss GPU for retrieval inference')
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
    parser.add_argument('--question_maxlength', type=int, default=128, help="Maximum number of tokens in a question")
    parser.add_argument('--random', type=int, default=0, help="random document")


    # 1024:
    parser.add_argument('--retrieved_max_length', type=int, default=256)
    parser.add_argument('--context_len', type=int, default=256)
    parser.add_argument('--pred_len', type=int, default=256)

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
    return parser.parse_args()


def compute_perplexity_data(model, indices=None, args=None):
    def save_prob(output):
        overall_output["all_logprobs"].append(output["logprobs"])
        overall_output["all_positions"].append(output["positions"])
        overall_output["aggregate_length"] += output["length"]
        overall_output["aggregate_utf8_length"] += output["utf8_length"]

    overall_output = {
        "all_logprobs": [],
        "all_positions": [],
        "aggregate_length": 0,
        "aggregate_utf8_length": 0.
    }
    if args.data.startswith("wikitext"):
        data = load_dataset("wikitext", args.data, split=f"test[0%:{int(args.data_ratio*100)}%]")
        data = data["text"]
        for i in tqdm(range(0, len(data), 10000)):
            batch = data[i:i+10000]
            doc = "\n\n".join(batch)
            output = model.get_perplexity_data(doc)
            if not output:
                continue
            save_prob(output)
    else:
        reader = lm_dataformat.Reader(args.data)
        embed()  # set ratio
        for i, doc in enumerate(tqdm_lib.tqdm(reader.stream_data())):
            if indices is not None and i not in indices:
                continue
            output = model.get_perplexity_data(doc)
            if not output:
                continue
            save_prob(output)
    return overall_output


def main():
    args = parse_args()
    model = models.create_model(args.model_config_path)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    model.context_len = args.context_len
    model.max_seq_len = args.context_len + args.pred_len
    if args.retrieved_max_length != 0:
        args.do_retrieval=1
    else:
        args.do_retrieval=0
    model.initialize_retriever(args)

    if args.doc_indices_path:
        assert args.max_docs is None
        indices = set(utils.read_json(args.doc_indices_path))
    elif args.max_docs:
        assert args.doc_indices_path is None
        indices = set(range(args.max_docs))
    else:
        indices = None

    perplexity_data = compute_perplexity_data(model=model, indices=indices, args=args)

    torch.save(perplexity_data, args.output_path)


if __name__ == "__main__":
    main()
