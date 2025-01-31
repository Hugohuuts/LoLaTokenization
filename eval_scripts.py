import torch
import json
import os

from tqdm import tqdm
from prediction_utilities import get_prediction
from custom_tokenizer_abstract import CustomTokenizerGeneral
from custom_models import load_custom_class
from argparse import ArgumentParser
from  method_mapping import *


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
    arg_parser.add_argument("--model", type=str, choices=["bart", "roberta", "minilm"])
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
        custom_tokenizer = CustomTokenizerGeneral(tokenizer_nli, tokenization_func, separator=model_parameters["separator_marker"], special_space_token=model_parameters["special_space_token"], max_length=model_parameters["max_length"])
        tokenizer_nli = custom_tokenizer

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
