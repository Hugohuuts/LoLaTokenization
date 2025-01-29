import torch
import json
import pandas as pd
import os

from tqdm import tqdm
from prediction_utilities import get_prediction
from example_tokenization import custom_tokenization
from tokenizers_pos import noun_tokenizer, verb_tokenizer, adjective_tokenizer
from custom_tokenizer_abstract import CustomTokenizerGeneral
from custom_models import load_custom_class
from argparse import ArgumentParser
from length_tokenizer import custom_tokenization_word_length, unigram_tokenizer, bigram_tokenizer, trigram_tokenizer
from greedy_tokenizer import greedy_prefix_tokenization, greedy_suffix_tokenization, greedy_longest_tokenization

TOK_METHOD_MAP = {
    "verb": verb_tokenizer,
    "noun": noun_tokenizer,
    "adj": adjective_tokenizer,
    "char": custom_tokenization,
    "custom_tokenization_word_length":custom_tokenization_word_length,
    "unigram_tokenizer":unigram_tokenizer,
    "bigram_tokenizer":bigram_tokenizer,
    "trigram_tokenizer":trigram_tokenizer,
    "greedy_prefix_tokenization": greedy_prefix_tokenization,
    "greedy_suffix_tokenization": greedy_suffix_tokenization,
    "greedy_longest_tokenization": greedy_longest_tokenization,
}

MODEL_MAP = {
    "roberta": {
        "model_link": "cross-encoder/nli-distilroberta-base",
        "separator_marker": "",
        "special_space_token": "Ġ"
    },
    "bert": {
        "model_link": "sentence-transformers/nli-bert-base",
        "separator_marker": "##",
        "special_space_token": ""
    },
    "minilm": {
        "model_link": "cross-encoder/nli-MiniLM2-L6-H768",
        "separator_marker": "",
        "special_space_token": "Ġ"
    }
}

def predict(premise, hypothesis, model_nli, tokenizer, is_custom=True, **tokenizer_args):
    if is_custom:
        input = (premise, hypothesis)
    else:
        input = premise + " " + hypothesis
    prediction = get_prediction(premise_hypothesis=input, model_nli=model_nli, custom_tokenizer=tokenizer, **tokenizer_args)
    
    return {"label": prediction["label"], "prob": prediction["prob"], "all_prob": prediction["all_probs"]}

def predict_loop(data, model_nli, tokenizer, is_custom=True, **tokenizer_args):
    results = []
    total = len(data["label"])
    for premise, hypothesis in tqdm(zip(data["sent1"], data["sent2"]), total=total):
        results += [predict(premise, hypothesis=hypothesis, model_nli=model_nli, tokenizer=tokenizer, is_custom=is_custom, **tokenizer_args)]

    return results

def save_results(results, file_name):
    with open(file_name, "w") as file:
        for idx, res in enumerate(results):
            output = {
                "id": idx,
                "result": res
            }
            json.dump(output, file)
            file.write("\n")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--path", type=str)
    arg_parser.add_argument("--tok_method", type=str, default="")
    arg_parser.add_argument("--model", type=str, choices=["bert", "roberta", "minilm"])
    arg_parser.add_argument("--do_custom", action="store_true")
    args = arg_parser.parse_args()

    if not args.tok_method and args.do_custom:
        print("You have to provide a tokenisation method if you want to use custom tokenisation!")
        exit(1)

    data_path = args.path
    model_parameters = MODEL_MAP[args.model]
    do_custom = args.do_custom

    model_args = {}
    tokenizer_nli, model_nli = load_custom_class(model_parameters["model_link"], device, **model_args)
    if do_custom:
        tokenization_func = TOK_METHOD_MAP[args.tok_method]
        custom_tokenizer = CustomTokenizerGeneral(tokenizer_nli, tokenization_func, separator=model_parameters["separator_marker"], special_space_token=model_parameters["special_space_token"])
        vocabulary_id2tok = {tok_id:tok for tok, tok_id in tokenizer_nli.vocab.items()}
        tokenizer_nli = custom_tokenizer

    # python eval_scripts.py --path "data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl" --tok_method adj --model roberta --do_custom true
    # python eval_scripts.py --path "data/snli_1.0/snli_1.0_test.jsonl" --tok_method adj --model roberta --do_custom true
    data = {
        "label": [],
        "sent1": [],
        "sent2": []
    }
    with open(data_path, "r") as file:
        for line in tqdm(file):
            aux_dict = json.loads(line)
            data["label"] += [aux_dict["gold_label"]]
            data["sent1"] += [aux_dict["sentence1"]] # premise
            data["sent2"] += [aux_dict["sentence2"]] # hypothesis

    tokenizer_args_normal = {
        "return_tensors": "pt"
    }
    tokenizer_args_custom = {
        "do_lowercase": True,
        "return_tensors": "pt"
    }
    tokenizer_args = tokenizer_args_custom if do_custom else tokenizer_args_normal

    # to see progress during operation: progress_apply instead of apply
    results = predict_loop(data, model_nli, tokenizer_nli, is_custom=do_custom, **tokenizer_args)

    os.makedirs(f"results/{args.model}/", exist_ok=True)
    file_name = f"results/{args.model}/{data_path.split('/')[-1][:-6]}-{args.tok_method if args.tok_method else 'standard'}-{args.model}.jsonl"
    
    save_results(results, file_name)