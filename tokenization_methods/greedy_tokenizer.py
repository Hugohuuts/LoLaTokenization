from typing import List, Tuple, Union

def greedy_prefix_tokenization(
    premise_hypothesis: Union[Tuple[str, str], List[str]],
    separator_marker: str = "",
    vocab: set = None,
    **tokenization_args
) -> Tuple[List[str], List[str]]:
    """Tokenizes text using longest prefix matching."""

    def _greedy_tokenize(text: str, max_length: int = 5, min_length: int = 2) -> List[str]:
        tokens = []
        words = text.strip().split()

        for word in words:
            current_pos = 0
            word_tokens = []

            while current_pos < len(word):
                best_token = word[current_pos:current_pos + min_length]
                best_length = min_length

                for length in range(max_length, min_length - 1, -1):
                    if current_pos + length <= len(word):
                        candidate = word[current_pos:current_pos + length]
                        if vocab is None or candidate in vocab:
                            best_token = candidate
                            best_length = length
                            break

                word_tokens.append(best_token)
                current_pos += best_length

            tokens.extend(word_tokens)

        return tokens

    max_length = tokenization_args.get('max_token_length', 5)
    min_length = tokenization_args.get('min_token_length', 2)

    return (
        _greedy_tokenize(premise_hypothesis[0], max_length, min_length),
        _greedy_tokenize(premise_hypothesis[1], max_length, min_length),
    )


def greedy_suffix_tokenization(
    premise_hypothesis: Union[Tuple[str, str], List[str]],
    separator_marker: str = "",
    vocab: set = None,
    **tokenization_args
) -> Tuple[List[str], List[str]]:
    """Tokenizes text using longest suffix matching."""

    def _greedy_tokenize(text: str, max_length: int = 5, min_length: int = 2) -> List[str]:
        tokens = []
        words = text.strip().split()

        for word in words:
            current_pos = len(word)
            word_tokens = []

            while current_pos > 0:
                best_token = word[max(0, current_pos - min_length):current_pos]
                best_length = min_length

                for length in range(max_length, min_length - 1, -1):
                    if current_pos - length >= 0:
                        candidate = word[current_pos - length:current_pos]
                        if vocab is None or candidate in vocab:
                            best_token = candidate
                            best_length = length
                            break

                if separator_marker and word_tokens:
                    best_token = f"{separator_marker}{best_token}"

                word_tokens.insert(0, best_token)
                current_pos -= best_length

            tokens.extend(word_tokens)

        return tokens

    max_length = tokenization_args.get('max_token_length', 5)
    min_length = tokenization_args.get('min_token_length', 2)

    return (
        _greedy_tokenize(premise_hypothesis[0], max_length, min_length),
        _greedy_tokenize(premise_hypothesis[1], max_length, min_length),
    )


def greedy_longest_tokenization(
    premise_hypothesis: Union[Tuple[str, str], List[str]],
    separator_marker: str = "",
    vocab: set = None,
    **tokenization_args
) -> Tuple[List[str], List[str]]:
    """
    Tokenizes text using the longest token match (whole word, prefix, or suffix).

    - If the whole word is valid (in vocab or within max length), keep it as is.
    - Otherwise, find the longest prefix or suffix match.

    Args:
        premise_hypothesis: Tuple or list containing (premise, hypothesis)
        separator_marker: Special character(s) used between tokens
        vocab: Set of known vocabulary words for validation
        tokenization_args: Additional tokenization arguments:
            - max_token_length: Maximum token length (default: 5)
            - min_token_length: Minimum token length (default: 2)

    Returns:
        Tuple containing (premise_tokens, hypothesis_tokens)
    """

    def _greedy_tokenize(text: str, max_length: int = 5, min_length: int = 2) -> List[str]:
        tokens = []
        words = text.strip().split()

        for word in words:
            # If the whole word is valid, keep it
            if (vocab is None or word in vocab) and len(word) <= max_length:
                tokens.append(word)
                continue

            current_pos = 0
            word_tokens = []

            while current_pos < len(word):
                best_token = word[current_pos:current_pos + min_length]
                best_length = min_length

                # Look for the longest valid prefix or suffix
                for length in range(max_length, min_length - 1, -1):
                    if current_pos + length <= len(word):
                        prefix_candidate = word[current_pos:current_pos + length]
                        suffix_candidate = word[len(word) - length:]

                        # Prefer the longest valid token (whole word > prefix > suffix)
                        if vocab is None or prefix_candidate in vocab:
                            best_token = prefix_candidate
                            best_length = length
                            break
                        if vocab is None or suffix_candidate in vocab:
                            best_token = suffix_candidate
                            best_length = length
                            break

                word_tokens.append(best_token)
                current_pos += best_length  # Move forward in the word

            tokens.extend(word_tokens)

        return tokens

    max_length = tokenization_args.get('max_token_length', 5)
    min_length = tokenization_args.get('min_token_length', 2)

    return (
        _greedy_tokenize(premise_hypothesis[0], max_length, min_length),
        _greedy_tokenize(premise_hypothesis[1], max_length, min_length),
    )
