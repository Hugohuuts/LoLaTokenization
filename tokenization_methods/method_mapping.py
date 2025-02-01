from tokenization_methods.character_split import custom_tokenization
from tokenization_methods.tokenizers_pos import noun_tokenizer, verb_tokenizer, adjective_tokenizer
from tokenization_methods.length_tokenizer import unigram_tokenizer
from tokenization_methods.greedy_tokenizer import greedy_prefix_tokenization, greedy_suffix_tokenization, greedy_longest_tokenization
from tokenization_methods.adverserial_tokenization import adverserial_shuffle_letters, adverserial_shuffle_tokens_one, adverserial_shuffle_tokens_one_neighbour, adverserial_pos_synonym_noun, adverserial_pos_synonym_verb

TOK_METHOD_MAP = {
    "verb": verb_tokenizer,
    "noun": noun_tokenizer,
    "adj": adjective_tokenizer,
    "char": custom_tokenization,
    "unigram_tokenizer":unigram_tokenizer,
    "greedy_prefix_tokenization": greedy_prefix_tokenization,
    "greedy_suffix_tokenization": greedy_suffix_tokenization,
    "greedy_longest_tokenization": greedy_longest_tokenization,
    "adverserial_shuffle_letters": adverserial_shuffle_letters, 
    "adverserial_shuffle_tokens_one": adverserial_shuffle_tokens_one, 
    "adverserial_shuffle_tokens_one_neighbour": adverserial_shuffle_tokens_one_neighbour,
    "adverserial_pos_synonym_noun": adverserial_pos_synonym_noun,
    "adverserial_pos_synonym_verb": adverserial_pos_synonym_verb,
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