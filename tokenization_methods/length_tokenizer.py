from typing import List, Tuple, Union

def unigram_tokenizer(premise_hypothesis: Union[Tuple[str, str], List[str]], separator_marker: str="", special_space_token: str="") -> Tuple[List[str], List[str]]:
    """
    Tokenizes text into unigrams (single words).
    """
    def _tokenize_unigrams(text: str) -> List[str]:
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            if i != 0:
                word = special_space_token + word
            tokens.append(word)
        return tokens

    premise_tokens = _tokenize_unigrams(premise_hypothesis[0])
    hypothesis_tokens = _tokenize_unigrams(premise_hypothesis[1])

    return premise_tokens, hypothesis_tokens

def bigram_tokenizer(premise_hypothesis: Union[Tuple[str, str], List[str]], separator_marker: str="", special_space_token: str="") -> Tuple[List[str], List[str]]:
    """
    Tokenizes text into bigrams (pairs of words).
    """
    def _tokenize_bigrams(text: str) -> List[str]:
        words = text.split()
        tokens = []
        for i in range(len(words) - 1):
            if i != 0:
                words[i] = special_space_token + words[i]
                words[i+1] = special_space_token + words[i+1]
            else:
                words[i+1] = special_space_token + words[i+1]
            tokens.append(words[i])
            tokens.append(words[i + 1])
        return tokens

    premise_tokens = _tokenize_bigrams(premise_hypothesis[0])
    hypothesis_tokens = _tokenize_bigrams(premise_hypothesis[1])

    return premise_tokens, hypothesis_tokens
