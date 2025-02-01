from typing import List, Tuple, Union
import math

def custom_tokenization_word_length(premise_hypothesis: Union[Tuple[str, str], List[str]], separator_marker: str="", special_space_token: str="", **tokenization_args) -> Tuple[List[str], List[str]]:
    """
    Custom tokenization method that returns separate tokens for premise and hypothesis.
    Compatible with CustomTokenizerGeneral class.

    Args:
        premise_hypothesis: Tuple or list containing (premise, hypothesis)
        separator_marker: Special character(s) used by tokenizers when splitting words
        tokenization_args: Additional tokenization arguments including:
            - lengths: List of n-gram lengths to use (default: [1, 2, 3])

    Returns:
        Tuple containing (premise_tokens, hypothesis_tokens)
    """
    def _tokenize_text(text: str, lengths: List[int] = [1, 2, 3]) -> List[str]:
        # Clean and split the text into words
        words = text.split()
        tokens = []

        # Generate tokens for each length
        for length in lengths:
            if length <= 0:
                continue
            # Loop through the words to create n-grams
            for i in range(len(words) - length + 1):
                if length == 1:
                    # For single words, just add space prefix if not first token
                    token = words[i]
                    if i > 0:
                        token = ' ' + token
                    tokens.append(token)
                else:
                    # For n-grams, add each word separately with appropriate space prefixes
                    for j, word in enumerate(words[i:i + length]):
                        if j == 0 and i == 0:
                            tokens.append(word)
                        else:
                            tokens.append(' ' + word)

        return tokens

    # Get lengths from tokenization_args if provided
    lengths = tokenization_args.get('lengths', [1, 2, 3])

    # Tokenize both premise and hypothesis
    premise_tokens = _tokenize_text(premise_hypothesis[0], lengths)
    hypothesis_tokens = _tokenize_text(premise_hypothesis[1], lengths)



def unigram_tokenizer(premise_hypothesis: Union[Tuple[str, str], List[str]], separator_marker: str="", special_space_token: str="") -> Tuple[List[str], List[str]]:
    """
    Tokenizes text into unigrams (single words).
    """
    def _tokenize_unigrams(text: str) -> List[str]:
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
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
            tokens.append(words[i])
            tokens.append(words[i + 1])
        return tokens

    premise_tokens = _tokenize_bigrams(premise_hypothesis[0])
    hypothesis_tokens = _tokenize_bigrams(premise_hypothesis[1])

    return premise_tokens, hypothesis_tokens
