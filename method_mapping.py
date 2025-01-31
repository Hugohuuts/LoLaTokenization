from example_tokenization import custom_tokenization
from tokenizers_pos import noun_tokenizer, verb_tokenizer, adjective_tokenizer
from length_tokenizer import custom_tokenization_word_length, unigram_tokenizer, bigram_tokenizer
from greedy_tokenizer import greedy_prefix_tokenization, greedy_suffix_tokenization, greedy_longest_tokenization

TOK_METHOD_MAP = {
    "verb": verb_tokenizer,
    "noun": noun_tokenizer,
    "adj": adjective_tokenizer,
    "char": custom_tokenization,
    "custom_tokenization_word_length":custom_tokenization_word_length,
    "unigram_tokenizer":unigram_tokenizer,
    "bigram_tokenizer":bigram_tokenizer,
    "greedy_prefix_tokenization": greedy_prefix_tokenization,
    "greedy_suffix_tokenization": greedy_suffix_tokenization,
    "greedy_longest_tokenization": greedy_longest_tokenization,
}

MODEL_MAP = {
    "roberta": {
        "model_link": "cross-encoder/nli-distilroberta-base",
        "separator_marker": "",
        "special_space_token": "Ġ",
        "max_length": 512
    },
    "bart": {
        "model_link": "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "separator_marker": "",
        "special_space_token": "Ġ",
        "max_length": 1024
    },
    "minilm": {
        "model_link": "cross-encoder/nli-MiniLM2-L6-H768",
        "separator_marker": "",
        "special_space_token": "Ġ",
        "max_length": 512
    }
}