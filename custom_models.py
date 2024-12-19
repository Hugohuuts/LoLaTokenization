import torch
from abc import ABC, abstractmethod
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import RobertaConfig, BertConfig, PreTrainedModel

class CustomModelAbstract(PreTrainedModel, ABC):
    """
    Abstract class serving as a template for custom models.
    """
    def __init__(self, model_name: str, **kwargs) -> None:
        config = AutoConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        super().__init__(config)
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)

    @abstractmethod
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, output_attentions = None, output_hidden_states = None, return_dict = None):
        pass

class CustomRoberta(CustomModelAbstract):
    config_class = RobertaConfig

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, output_attentions = None, output_hidden_states = None, return_dict = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.model.classifier(sequence_output)

        loss = None
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CustomBert(CustomModelAbstract):
    config_class = BertConfig

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, output_attentions = None, output_hidden_states = None, return_dict = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)

        loss = None
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def load_custom_class(model_name_or_repo_link: str, device: torch.device=None, **model_args) -> tuple[AutoTokenizer, CustomModelAbstract]:
    """
    Wrapper method to automatically load (or download) a given custom model.
    ---
    Args: 
        model_name_or_repo_link: name of the model to be loaded, in the `HuggingFace` format: `<author>/<model_name>`.
        device: on which device to load the model; if not given, automatically detects the best option.
        model_args: additional arguments for model loading, e.g. using only specific hidden layers.
    Returns:
        The pre-trained tokenizer and the custom model.
    """
    class_type = None

    # if available, load the model on the GPU
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if "roberta" in model_name_or_repo_link.lower():
        class_type = CustomRoberta
    elif "bert" in model_name_or_repo_link.lower() and "roberta" not in model_name_or_repo_link.lower():
        class_type = CustomBert
    
    assert class_type != None, "Could not find a class with that name."

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_repo_link)
    model = class_type(model_name_or_repo_link, **model_args).to(device)

    return tokenizer, model