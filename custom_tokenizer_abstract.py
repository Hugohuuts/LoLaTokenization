from transformers import AutoTokenizer, BatchEncoding
import torch
import warnings

class MissingTokenizationFunctionError(Exception):
    pass

class CustomTokenizerGeneral:
    def __init__(self, tokenizer: AutoTokenizer, tokenization_func: callable=None, separator: str="##", device: torch.device=None, special_space_token: str="Ġ", max_length: int=512):
        """
        Template class for calling a custom tokenization method.
        ---
        Args:
            tokenizer: any pre-trained tokenizer.
            tokenization_func: a custom tokenization method that **must** return a tuple or list of the tokenized premises and hypothesis respectively.
            separator: special character(s) use by tokenizers when splitting words (e.g., `##` for BERT).
            device: on which device to move the tensors; if not given, automatically detects the best option.
            special_space_token: character used as replacement for leading spaces of tokens; RoBERTa uses `Ġ` instead of spaces while BERT does not consider spaces (so ``).
        """
        self.vocabulary = tokenizer.vocab
        self.original_tokenizer = tokenizer
        self.special_space_token = special_space_token
        self.max_length = max_length
        self.valid_tokens = set(self.vocabulary.keys())
        self.special_tokens_set = set(tokenizer.special_tokens_map.values())
        self.original_tokenizer_name = tokenizer.name_or_path.lower()
        self.vocabulary_id2tok = {tok_id:tok for tok, tok_id in self.vocabulary.items()}

        self._load_special_tokens(self.original_tokenizer_name, tokenizer)

        self.tokenization_func = tokenization_func
        self.separator = separator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device
        print(f"Tensors and operations will be done on {self.device}.")

    def _load_special_tokens(self, tokenizer_name: str, tokenizer: AutoTokenizer) -> None:
        """
        Setter function to load special tokens, for different models.
        ---
        Args:
            tokenizer_name: name of the given pre-trained tokenizer.
            tokenizer: any pre-trained tokenizer.
        """
        if "bert" in tokenizer_name and "roberta" not in tokenizer_name:
            self.unk_token_id, self.unk_token = self.vocabulary[tokenizer.unk_token], tokenizer.unk_token
            self.sep_token_id, self.sep_token = self.vocabulary[tokenizer.sep_token], tokenizer.sep_token
            self.pad_token_id, self.pad_token = self.vocabulary[tokenizer.pad_token], tokenizer.pad_token
            self.class_token_id, self.class_token = self.vocabulary[tokenizer.cls_token], tokenizer.cls_token
        else:
            self.unk_token_id, self.unk_token = self.vocabulary[tokenizer.unk_token], tokenizer.unk_token
            self.sep_token_id, self.sep_token = self.vocabulary[tokenizer.sep_token], tokenizer.sep_token
            self.pad_token_id, self.pad_token = self.vocabulary[tokenizer.pad_token], tokenizer.pad_token
            self.bos_token_id, self.bos_token = self.vocabulary[tokenizer.bos_token], tokenizer.bos_token
            self.eos_token_id, self.eos_token = self.vocabulary[tokenizer.eos_token], tokenizer.eos_token

    def replace_prefix_space_with_special(self, token_list: list[str]) -> list[str]:
        """
        Replace leading spaces with special space character.
        ---
        Args:
            token_list: list of `string` tokens.
        Returns:
            List of string tokens with leading special space character (e.g., `Ġcat`)
        """
        if self.special_space_token == " ":
            return token_list
        
        return [tok.replace(" ", self.special_space_token) for tok in token_list]

    def get_token_id(self, token: str) -> int:
        """
        Get the ID of a token from the vocabulary.
        ---
        Args:
            token: string token.
        Returns:
            Integer ID.
        """
        return_token = None
        if token in self.valid_tokens:
            return_token = self.vocabulary[token]
        else:
            aux_token = token
            # print(token, self.special_space_token)
            if self.special_space_token != " " and token[0] == self.special_space_token:
                aux_token = token.replace(self.special_space_token, " ") # original tokenizer only accepts ' ' and not the special character, as is leads to buggy outputs

            original_tok_split = self.original_tokenizer.encode(aux_token)[1:-1] # remove the extra bos and eos tokens
            decoded_original_tok = [self.original_tokenizer.decode(original_tok) for original_tok in original_tok_split]
            # warnings.warn(f"\033[31mWarning: '{token}' was not found in the original tokenizer's vocabulary; '{token}' will be split into '{decoded_original_tok}' using the original tokenizer.\033[0m", stacklevel=2)
            return_token = original_tok_split 

        if isinstance(return_token, int):
                return_token = [return_token]
        return return_token
    
    def encode(self, token_list: list[str]) -> list:
        """
        Encode an entire list of string tokens.
        ---
        Args:
            token_list: list of string tokens.
        Returns:
            List of integer token IDs.
        """
        aux_token_list = []
        for token in token_list:
            aux_token_list += self.get_token_id(token)

        return aux_token_list

    def get_offset_mapping(self, token_list):
        running_length = 0
        spans = []
        for tok in token_list:
            start_idx = running_length
            end_idx = running_length + len(tok)
            if tok in self.special_tokens_set:
                spans += [(0,0)]
                continue
            if tok[0] == self.special_space_token:
                start_idx += 1
            spans += [(start_idx, end_idx)]
            running_length = end_idx

        return spans
    
    def get_attention_mask(self, token_id_list):
        return [int(token_id != self.pad_token_id) for token_id in token_id_list]

    def decode(self, token_list: list[int], remove_special_markings: bool=False) -> str:
        """
        Decode a given list of token IDs to their string representation.
        ---
        Args:
            token_list: list of token IDs.
            remove_special_markings: whether to remove the special space marker
        Returns:
            String of the original tokens.
        """
        return " ".join([self.vocabulary_id2tok[token] if not remove_special_markings else self.vocabulary_id2tok[token].replace(self.special_space_token, "") for token in token_list])

    def _pad_sequence(self, token_list: list[int], padding_limit: int) -> list[int]:
        """
        Pad a list of token IDs **to the right**.
        ---
        Args:
            token_list: list of token IDs.
            padding_limit: how much padding to be added.
        Returns:
            List of tokens ID padded to the right with padding token IDs.
        """
        while len(token_list) < padding_limit:
            token_list += [self.pad_token_id]

        return token_list

    def combine_token_list(self, tok_list_1: list[str], tok_list_2: list[str]) -> list[str]:
        """
        Combine two token lists based on model expected input.
        ---
        Args:
            tok_list_1: list of token IDs.
            tok_list_2: list of token IDs.
        Returns:
            List of token IDs interlaced with model-specific special tokens.
        """

        combined_list = None
        overflow_length = len(tok_list_1) + len(tok_list_2)
        # print(len(tok_list_1), len(tok_list_2), overflow_length)
        if "roberta" in self.original_tokenizer_name or "minilm" in self.original_tokenizer_name:
            num_markers = 4
        elif "bert" in self.original_tokenizer_name and "roberta" not in self.original_tokenizer_name:
            num_markers = 3
        elif "bart" in self.original_tokenizer_name:
            num_markers = 4

        # print(overflow_length, len(tok_list_1), len(tok_list_2))
        if (overflow_length + num_markers) - self.max_length > 0:
            # max_length < 514 (bcs of end and start)
            # 516 = 512 + 2 (bos, eos) + 2 (separators between inputs)
            # e.g. tokl1 = 10, tokl2 = 15 ==> total=25, max=17 (15+2) ==> overflow=8 ==>tokl2 = 15-8=7
            # overflow_length = 11, tok_limit = 5 -> tok_list_1 = 6, tok_list_2 = 5
            # overflow_length = 12, tok_limit = 6 -> tok_list_1 = 6, tok_list_2 = 6
            #overflow_length = 13, tok_limit = 6 -> tok_list_1 = 7, tok_list_2 = 6
            tok_limit_1 = self.max_length // 2 - 1
            tok_limit_2 = self.max_length // 2 - 1 - (num_markers // 2)
            tok_list_1 = tok_list_1[:tok_limit_1]
            tok_list_2 = tok_list_2[:tok_limit_2]

        if "bert" in self.original_tokenizer_name and "roberta" not in self.original_tokenizer_name:
            combined_list = [self.class_token] + tok_list_1 + [self.sep_token] + tok_list_2 + [self.sep_token]
        else:
            combined_list = [self.bos_token] + tok_list_1 + [self.eos_token] + [self.sep_token] + tok_list_2 + [self.eos_token]

        assert len(combined_list) <= self.max_length, f"Too many tokens: {len(combined_list)}"

        return combined_list
    
    # remove any empty strings, if present
    def sanity_check(self, tok_list):
        return [tok for tok in tok_list if tok != ""]

    def __call__(self, premise_hypothesis: tuple[str]|list[str], **tokenization_args) -> dict:
        """
        This method should output model-ready values, in the form of a dictionary with torch tensors for input IDs and attention mask
        output for the input ids should be a nested list in the following form [[<bos> P1. P2. P3 ... Pn <eos> H <eos> ]]
        output for the attention ids should be a nested list of the following form [[1,1,1,1,1,1,1.....]] with 1 being tokens the model
        should pay attention to and 0 where the model will not be attentive to
        ---
        Args:
            premise_hypothesis: list or tuple of two elements containing premises in the first position and hypothesis in the second position.
            tokenization_args: any additional tokenization arguments that are legal for pre-trained tokenizers, e.g. `max_length`, `truncation`, `padding`.
        Returns:
            Dictionary that is transformer-model ready, contains `input_ids` and `attention_mask`.
        Raises:
            MissingTokenizationFunctionError if no tokenization function is provided.
        """
        # tokenization args: truncation, max_length, on how many characters to split, etc.
        if not self.tokenization_func:
            raise MissingTokenizationFunctionError
        
        do_lowercase = tokenization_args.pop("do_lowercase", False)
        if do_lowercase:
            premise_hypothesis = [text.lower() for text in premise_hypothesis]

        tokens_premises, tokens_hypothesis = self.tokenization_func(premise_hypothesis, self.separator, self.special_space_token)
        tokens_premises, tokens_hypothesis = self.sanity_check(tokens_premises), self.sanity_check(tokens_hypothesis)
        tokens_premises, tokens_hypothesis = self.replace_prefix_space_with_special(tokens_premises), self.replace_prefix_space_with_special(tokens_hypothesis)

        tokens = self.combine_token_list(tokens_premises, tokens_hypothesis)
        input_ids = self.encode(tokens)

        unk_tok_num = sum([token_id == self.unk_token_id for token_id in input_ids]) # not working always? some data simply is not available to models, e.g. ˨ˠ3 in mnli
        # assert not(unk_tok_num > 0), f"There are unknown tokens in the text! {unk_tok_num} unknown tokens."
        if unk_tok_num > 0:
            warnings.warn(f"There are {unk_tok_num} unknown tokens!")

        attention_mask = [int(token_id != self.pad_token_id) for token_id in input_ids]

        # input_ids on cuda
        # attention mask on cuda
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask # always 1 except for padding tokens
        }

        if tokenization_args.pop("return_tensors", False) == "pt":
            for key, val in output.items():
                if not isinstance(val, torch.Tensor):
                    aux_t = torch.as_tensor(val)
                    if len(aux_t.shape) > 1:
                        output[key] = aux_t
                    else:
                        output[key] = torch.as_tensor([val])

        return BatchEncoding(output)

#### example of how a new custom tokenization function can be applied
if __name__ == "__main__":
    from tokenization_methods.character_split import custom_tokenization

    repo_link_nli = "cross-encoder/nli-distilroberta-base"
    separator = ""
    special_space_token = "Ġ"

    # repo_link_nli = "sentence-transformers/nli-bert-base"
    # separator = "##"
    # special_space_token = ""

    tokenizer_nli = AutoTokenizer.from_pretrained(repo_link_nli)
    test_nli = [
        ("The cat is large. An ant is small.", "The cat is bigger than the ant.", "entailment"),
        ("Dumbo is by the tree and is not small. Bambi is an animal that is not large and is hungry.", "Dumbo is not smaller than Bambi.", "neutral")
        ]
    
    ### EXAMPLE
    print("Example tokenization")
    custom_general_tokenizer = CustomTokenizerGeneral(tokenizer_nli, custom_tokenization, separator=separator, special_space_token=special_space_token)
    for problem in test_nli:
        toks = custom_general_tokenizer(problem)
        aux_list = toks["input_ids"][0].detach().cpu().numpy()
        print(custom_general_tokenizer.decode(aux_list, remove_special_markings=False))
