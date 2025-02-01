import re
import numpy as np
from nltk.corpus import wordnet
import spacy

NLP = spacy.load("en_core_web_sm")

# adverserial_shuffle_letters
def adverserial_shuffle_letters(premises_hypothesis: tuple[str]|list[str], separator_marker: str="", space_marker: str=" ", **tokenization_args) -> tuple[list, list]:
    def _tokenization_granular(text: str, separator_marker: str, **tokenization_args) -> list[str]:
        text = re.sub(" {2,}", " ", text)
        text = re.split(r"( ?[\w]+)([!\"\#$%&\'()*+,-\.\/:;<=>?@[\\\]^_`{|}~])?", text)
        text = [tok for tok in text if tok not in  ("", " ", None)]
        # to_shuffle_text = set(random.sample(text, round(len(text) * 0.2))) # choose randomly 20% of the tokens to shuffle their letters
        tok_list = []
        for tok in text:
            tok = tok.replace(" ", space_marker)
            if np.random.random() <= 0.2:
                if tok[0] == space_marker and len(tok) > 2:
                    tok = tok[1:]
                    tok = list(tok)
                    np.random.shuffle(tok)
                    tok = space_marker + "".join(tok)
                elif len(tok) > 1:
                    tok = list(tok)
                    np.random.shuffle(tok)
                    tok = "".join(tok)

            tok_list += [tok]

        return tok_list
    
    tokens_premises = _tokenization_granular(premises_hypothesis[0], separator_marker, **tokenization_args)
    tokens_hypothesis = _tokenization_granular(premises_hypothesis[1], separator_marker, **tokenization_args)

    return tokens_premises, tokens_hypothesis

# shuffle tokens
def adverserial_shuffle_tokens_one(premises_hypothesis: tuple[str]|list[str], separator_marker: str="", space_marker: str=" ", **tokenization_args) -> tuple[list, list]:
    def _tokenization_granular(text: str, separator_marker: str, **tokenization_args) -> list[str]:
        text = re.sub(" {2,}", " ", text)
        text = re.split(r"( ?[\w]+)([!\"\#$%&\'()*+,-\.\/:;<=>?@[\\\]^_`{|}~])?", text)
        text = [tok for tok in text if tok not in  ("", " ", None)]
        text = np.asarray(text)
        if len(text) > 1:
            swap_idx = np.random.choice(range(len(text)), 2, replace=False)
            text[swap_idx] = text[swap_idx[::-1]] # swap tokens

        tok_list = []
        for tok in text:
            tok = tok.replace(" ", space_marker)
            tok_list += [tok]# unpack into letters

        return tok_list
    
    tokens_premises = _tokenization_granular(premises_hypothesis[0], separator_marker, **tokenization_args)
    tokens_hypothesis = _tokenization_granular(premises_hypothesis[1], separator_marker, **tokenization_args)

    return tokens_premises, tokens_hypothesis

def adverserial_shuffle_tokens_one_neighbour(premises_hypothesis: tuple[str]|list[str], separator_marker: str="", space_marker: str=" ", **tokenization_args) -> tuple[list, list]:
    def _tokenization_granular(text: str, separator_marker: str, **tokenization_args) -> list[str]:
        text = re.sub(" {2,}", " ", text)
        text = re.split(r"( ?[\w]+)([!\"\#$%&\'()*+,-\.\/:;<=>?@[\\\]^_`{|}~])?", text)
        text = [tok for tok in text if tok not in  ("", " ", None)]
        swap_idx_left = np.random.randint(0, len(text))
        swap_idx_right = (swap_idx_left + 1) % len(text) # in order to prevent indices outside array bounds, we just wrap around to the beginning
        text[swap_idx_left], text[swap_idx_right] = text[swap_idx_right], text[swap_idx_left]

        tok_list = []
        for tok in text:
            tok = tok.replace(" ", space_marker)
            tok_list += [tok]

        return tok_list
    
    tokens_premises = _tokenization_granular(premises_hypothesis[0], separator_marker, **tokenization_args)
    tokens_hypothesis = _tokenization_granular(premises_hypothesis[1], separator_marker, **tokenization_args)

    return tokens_premises, tokens_hypothesis


def adverserial_pos_synonym(premises_hypothesis: tuple[str]|list[str], separator_marker: str="", space_marker: str=" ", pos_list: list=["NOUN", "PROPN"], **tokenization_args) -> tuple[list, list]:
    def find_synonym(token):
        synonym_list = wordnet.synonyms(token)
        for syn in synonym_list:
            if syn:
                return syn[0].replace("_", " ")
        return token

    def change_synonym(text, pos_list):
        pos_tokens = NLP(text)
        aux_tok_list = []
        for idx, token in enumerate(pos_tokens):
            if token.pos_ in pos_list:
                token = find_synonym(token.text)
                token = space_marker + token if idx > 0 else token
                aux_tok_list += [token]
            else:
                token = space_marker + token.text if token.pos_ != "PUNCT" and idx > 0 else token.text
                aux_tok_list += [token]

        # aux_tok_list = "".join(aux_tok_list)
        return aux_tok_list
    
    tokens_premises = change_synonym(premises_hypothesis[0], pos_list, **tokenization_args)
    tokens_hypothesis = change_synonym(premises_hypothesis[1], pos_list, **tokenization_args)

    return tokens_premises, tokens_hypothesis

def adverserial_pos_synonym_noun(premises_hypothesis: tuple[str]|list[str], separator_marker: str="", space_marker: str=" ", **tokenization_args) -> tuple[list, list]:
    return adverserial_pos_synonym(premises_hypothesis, separator_marker=separator_marker, space_marker=space_marker, pos_list=["NOUN", "PROPN"], **tokenization_args)

def adverserial_pos_synonym_verb(premises_hypothesis: tuple[str]|list[str], separator_marker: str="", space_marker: str=" ", **tokenization_args) -> tuple[list, list]:
    return adverserial_pos_synonym(premises_hypothesis, separator_marker=separator_marker, space_marker=space_marker, pos_list=["VERB"], **tokenization_args)