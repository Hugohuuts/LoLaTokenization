import argparse
import json
import os
import numpy as np

from openai import OpenAI
from openai.types import Completion
from datasets import load_dataset
from math import ceil
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
from env_utils import load_env_from_file

MAX_CONTEXT = 4096

import torch

model_map = {
    "llama2": "meta-llama/Llama-2-7b-hf"
}

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return tokenizer

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument("--model", type=str)
    args_parser.add_argument("--port", type=str)
    args = args_parser.parse_args()

    model_name = model_map[args.model]
    port = args.port

    # llm, tokenizer = load_model_tokenizer(model_name)
    tokenizer = load_tokenizer(model_name)

    # system_prompt_len = len(tokenizer.encode(SYSTEM_PROMPT))
    # base_length = MAX_OUTPUT_LENGTH + SAFETY_MARGIN + system_prompt_len

    # openai_client =  OpenAI(api_key="NONE", base_url= f"http://localhost:{port}/v1")

    # current_prompt_length = base_length + len(tokenizer.encode(prompt_template))

    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {"role": "user", "content": prompt}
    # ]

    # response = openai_client.chat.completions.create(
    #     model=model_name,
    #     messages=messages,
    #     # max_tokens=MAX_OUTPUT_LENGTH,
    #     temperature=0.0,
    #     presence_penalty=0,
    #     frequency_penalty=0
    # )

    # torch.cuda.empty_cache()

    # json.dump({"system_prompt": SYSTEM_PROMPT, "prompt": prompt, "conversation": dict(response.choices[0].message), "usage": dict(response.usage)}, open(path_response + f"{idx}.json", "w"), indent=2)

    # time.sleep(10)