import torch
import numpy as np

def get_prediction(premise_hypothesis: tuple[str]|list[str], model_nli: callable, custom_tokenizer: callable, **tokenization_args) -> dict:
    """
    Wrapper method to obtain entailments labels given an input, a model, tokenizer, and optional tokenization arguments.
    ---
    Args:
        premise_hypothesis: list or tuple of two elements containing premises in the first position and hypothesis in the second position.
        model_nli: transformer model (of general type `AutoModelForSequenceClassification`) that can classify inferences.
        custom_tokenizer: tokenizer (either custom or pre-trained with a model) that must return a dictionary with `input_ids` and `attention_mask` as its items.
        tokenization_args: any additional tokenization arguments that are legal for pre-trained tokenizers, e.g. `max_length`, `truncation`, `padding`.
    Returns:
        A dictionary containing the most likely entailment `label` of the input, its `prob`, and the softmax distribution over `all_probs` .
    """
    tok_output = custom_tokenizer(premise_hypothesis, **tokenization_args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    tok_output.to(device)
    # print(tok_output, tok_output["input_ids"].shape, tok_output["attention_mask"].shape)
    model_outputs = model_nli(**tok_output)
    probs = torch.softmax(model_outputs.logits, dim=1).tolist()[0]
    output = {
        "label": model_nli.config.id2label[np.argmax(probs)],
        "prob": np.max(probs),
        "all_probs": {model_nli.config.id2label[prob_idx]: prob_aux for prob_idx, prob_aux in zip(np.argsort(probs)[::-1], np.sort(probs)[::-1])}
    }
    
    return output