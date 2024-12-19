import re

### EXAMPLE: creating a custom method and passing it along to the general CustomTokenizerGeneral class
def custom_tokenization(premises_hypothesis: tuple[str]|list[str], separator_marker: str="", **tokenization_args) -> tuple[list, list]:
    """
    Wrapper for the second example of custom tokenization.
    This method is simply passed to a custom tokenizer class and executed when the tokenizer is called.
    ---
    Args:
        premise_hypothesis: list or tuple of two elements containing premises in the first position and hypothesis in the second position.
        separator_marker: special character(s) use by tokenizers when splitting words (e.g., `##` for BERT).
        tokenization_args: any additional tokenization arguments that are legal for pre-trained tokenizers, e.g. `max_length`, `truncation`, `padding`.
    Returns:
        A tuple of lists containing the tokens of the premises in the first slot and the hypothesis in the second slot.
    """
    def _custom_tokenization_granular(text: str, separator_marker: str, **tokenization_args) -> list[str]:
        """
        Split a string into tokens such that the leading whitespaces are also part of the tokens.
        ---
        Args:
            text: input string, consisting of **one sentence**.
            separator_marker: special character(s) use by tokenizers when splitting words (e.g., `##` for BERT).
            tokenization_args: any additional tokenization arguments that are legal for pre-trained tokenizers, e.g. `max_length`, `truncation`, `padding`.
        Returns:
            List of string tokens
        """
        text = re.split(r"( [\w]+)", text)
        text = [tok for tok in text if tok != ""]
        tok_list = []
        for tok in text:
            tok_list += [tok[0] if tok[0] != " " else tok[:2]] # special case for the first words of a string
            start_idx = 2 if tok[0] == " " else 1
            for chr in tok[start_idx:]:
                tok_list += [f"{separator_marker}{chr}"]

        return tok_list
    
    tokens_premises = _custom_tokenization_granular(premises_hypothesis[0], separator_marker, **tokenization_args)
    tokens_hypothesis = _custom_tokenization_granular(premises_hypothesis[1], separator_marker, **tokenization_args)

    return tokens_premises, tokens_hypothesis