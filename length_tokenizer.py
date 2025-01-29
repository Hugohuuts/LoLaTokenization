from typing import List, Tuple, Union
import math

def custom_tokenization_word_length(premise_hypothesis: Union[Tuple[str, str], List[str]], separator_marker: str="", **tokenization_args) -> Tuple[List[str], List[str]]:
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

    return premise_tokens, hypothesis_tokens

def custom_char_tokenization(premise_hypothesis: Union[Tuple[str, str], List[str]], separator_marker: str="", **tokenization_args) -> Tuple[List[str], List[str]]:
    """
    Custom character-level tokenization method.

    Args:
        premise_hypothesis: Tuple or list containing (premise, hypothesis)
        separator_marker: Special character(s) used by tokenizers when splitting words
        tokenization_args: Additional tokenization arguments

    Returns:
        Tuple containing (premise_char_tokens, hypothesis_char_tokens)
    """
    def _tokenize_characters(text: str, lengths: List[int] = [2, 3]) -> List[str]:
        text = text.replace(" ", "")
        tokens = []

        for length in lengths:
            if length <= 0:
                continue
            for i in range(len(text) - length + 1):
                token = text[i:i + length]
                if separator_marker and i > 0:
                    token = f"{separator_marker}{token}"
                tokens.append(token)

        return tokens

    # Get lengths from tokenization_args if provided
    lengths = tokenization_args.get('lengths', [2, 3])

    # Tokenize both premise and hypothesis
    premise_tokens = _tokenize_characters(premise_hypothesis[0], lengths)
    hypothesis_tokens = _tokenize_characters(premise_hypothesis[1], lengths)

    return premise_tokens, hypothesis_tokens


def golden_chunk_tokenization(
    premise_hypothesis: Union[Tuple[str, str], List[str]],
    separator_marker: str="",
    special_space_token: str = " ",
    **tokenization_args
) -> Tuple[List[str], List[str]]:
    """
    Golden-Chunk tokenization method that uses increasing chunk sizes based on the golden ratio.

    Args:
        premise_hypothesis: Tuple or list containing (premise, hypothesis)
        separator_marker: Special character(s) used by tokenizers when splitting words
        special_space_token: Special space token to use in the tokenization
        tokenization_args: Additional arguments including:
            - initial_length: Starting chunk size (default: 2)
            - ratio: Growth ratio (default: 1.618)
            - max_chunk_length: Maximum chunk size (default: 8)
            - rounding_mode: How to round chunk sizes ('floor', 'ceil', 'round') (default: 'floor')
            - reset_at_max: Whether to reset chunk size after hitting max (default: True)

    Returns:
        Tuple containing (premise_tokens, hypothesis_tokens)
    """
    def _golden_chunk_text(text: str, **args) -> List[str]:
        # Initialize parameters
        initial_length = args.get('initial_length', 2)
        ratio = args.get('ratio', 1.618)
        max_chunk_length = args.get('max_chunk_length', 8)
        rounding_mode = args.get('rounding_mode', 'floor')
        reset_at_max = args.get('reset_at_max', True)

        tokens = []
        i = 0
        current_length = initial_length
        text = text.replace(" ", "")  # Remove spaces

        while i < len(text):
            # Extract chunk
            chunk = text[i:i + current_length]
            if chunk:  # Only add non-empty chunks
                tokens.append(chunk)

            # Move pointer
            i += current_length

            # Calculate next chunk length
            next_length = current_length * ratio
            if rounding_mode == 'floor':
                next_length = int(next_length)
            elif rounding_mode == 'ceil':
                next_length = math.ceil(next_length)
            else:  # 'round'
                next_length = round(next_length)

            # Apply maximum length constraint
            if next_length > max_chunk_length:
                current_length = initial_length if reset_at_max else max_chunk_length
            else:
                current_length = max(1, next_length)  # Ensure at least length 1

        return tokens

    # Tokenize both premise and hypothesis
    premise_tokens = _golden_chunk_text(premise_hypothesis[0], **tokenization_args)
    hypothesis_tokens = _golden_chunk_text(premise_hypothesis[1], **tokenization_args)

    return premise_tokens, hypothesis_tokens

def unigram_tokenizer(
    premise_hypothesis: Union[Tuple[str, str], List[str]],
    separator_marker: str="",
    special_space_token: str = " "
) -> Tuple[List[str], List[str]]:
    """
    Tokenizes text into unigrams (single words).
    """
    def _tokenize_unigrams(text: str) -> List[str]:
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            # Don't add space prefix - let CustomTokenizerGeneral handle it
            tokens.append(word)
        return tokens

    premise_tokens = _tokenize_unigrams(premise_hypothesis[0])
    hypothesis_tokens = _tokenize_unigrams(premise_hypothesis[1])

    return premise_tokens, hypothesis_tokens

def bigram_tokenizer(
    premise_hypothesis: Union[Tuple[str, str], List[str]],
    separator_marker: str="",
    special_space_token: str = " "
) -> Tuple[List[str], List[str]]:
    """
    Tokenizes text into bigrams (pairs of words).
    """
    def _tokenize_bigrams(text: str) -> List[str]:
        words = text.split()
        tokens = []
        for i in range(len(words) - 1):
            if i == 0:
                tokens.append(words[i])
            else:
                tokens.append(f"{special_space_token}{words[i]}")
            tokens.append(f"{special_space_token}{words[i + 1]}")
        return tokens

    premise_tokens = _tokenize_bigrams(premise_hypothesis[0])
    hypothesis_tokens = _tokenize_bigrams(premise_hypothesis[1])

    return premise_tokens, hypothesis_tokens

def trigram_tokenizer(
    premise_hypothesis: Union[Tuple[str, str], List[str]],
    separator_marker: str="",
    special_space_token: str = " "
) -> Tuple[List[str], List[str]]:
    """
    Tokenizes text into trigrams (triplets of words).
    """
    def _tokenize_trigrams(text: str) -> List[str]:
        words = text.split()
        tokens = []
        for i in range(len(words) - 2):
            if i == 0:
                tokens.append(words[i])
            else:
                tokens.append(f"{special_space_token}{words[i]}")
            tokens.append(f"{special_space_token}{words[i + 1]}")
            tokens.append(f"{special_space_token}{words[i + 2]}")
        return tokens

    premise_tokens = _tokenize_trigrams(premise_hypothesis[0])
    hypothesis_tokens = _tokenize_trigrams(premise_hypothesis[1])

    return premise_tokens, hypothesis_tokens
