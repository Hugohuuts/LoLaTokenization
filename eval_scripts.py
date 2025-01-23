import torch
import json
import pandas as pd

from tqdm import tqdm
from prediction_utilities import get_prediction
from example_tokenization import custom_tokenization
from tokenizers_pos import noun_tokenizer
from custom_tokenizer_abstract import CustomTokenizerGeneral
from custom_models import load_custom_class
from argparse import ArgumentParser

if __name__ == "__main__":
    # sub class from modelling_bert "RobertaForSequenceClassification" and override the forward method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--path", type=str)
    args = arg_parser.parse_args()

    repo_link_nli = "cross-encoder/nli-distilroberta-base"
    # repo_link_nli = "sentence-transformers/nli-bert-base"

    model_args = {}

    tokenizer_nli, model_nli = load_custom_class(repo_link_nli, device, **model_args)
    vocabulary_id2tok = {tok_id:tok for tok, tok_id in tokenizer_nli.vocab.items()}

    # BERT
    # custom_tokenizer = CustomTokenizerGeneral(tokenizer_nli, custom_tokenization, separator_marker="##", special_space_token="")
    # RoBERTa
    custom_tokenizer = CustomTokenizerGeneral(tokenizer_nli, noun_tokenizer, separator_marker="", special_space_token="Ġ")

    import json
    import pandas as pd

    # data_path = "data/" + "/snli_1.0" + "/snli_1.0_test.jsonl"
    # data_path = "data/" + "/multinli_1.0" + "/multinli_1.0_dev_mismatched.jsonl"
    data_path = args.path

    data = []
    limit = 500_000
    with open(data_path, "r") as file:
        for _ in range(limit):
            json_obj = file.readline()
            if json_obj != "":
                data += [json.loads(json_obj)]
            else:
                break


    tokenizer_args_normal = {
        "return_tensors": "pt"
    }
    tokenizer_args_custom = {
        "do_lowercase": True
    }

    responses = {
        "custom": [],
        "normal": []
    }

    data_df = {
    "label": [],
    "sent1": [],
    "sent2": []
}

    for datum in data:
        data_df["label"] += [datum["gold_label"]]
        data_df["sent1"] += [datum["sentence1"]] # premise
        data_df["sent2"] += [datum["sentence2"]] # hypothesis

    data_df = pd.DataFrame(data_df).iloc[:]
    print(data_df.shape)
    data_df.head()

    def df_predict(row, model_nli, tokenizer,  is_custom=True, **tokenizer_args):
        if is_custom:
            input = (row["sent1"], row["sent2"])
        else:
            input = row["sent1"] + " " + row["sent2"]
        prediction = get_prediction(input, model_nli, tokenizer, **tokenizer_args)
        
        return prediction["label"], prediction["prob"]

    tqdm.pandas()
    # to see progress during operation: progress_apply instead of apply
    results_custom = data_df.apply(df_predict, axis=1, model_nli=model_nli, tokenizer=custom_tokenizer, is_custom=True, **tokenizer_args_custom)

    results_custom.to_json('results_custom_tokenizer.json')
    # results_normal = data_df.apply(df_predict, axis=1, model_nli=model_nli, tokenizer=tokenizer_nli, is_custom=False, **tokenizer_args_normal)