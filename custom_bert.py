import torch
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertConfig, PreTrainedModel
from transformers import RobertaForSequenceClassification, RobertaConfig

class CustomBert(PreTrainedModel):
    config_class = RobertaConfig
    def __init__(self, config, **kwargs):
        # self.from_config(config)
        super().__init__(config)
        self._load_bert(kwargs["model_name"])

    def from_pretrained(**kwargs):
        super(**kwargs)

    def _load_bert(self, model_name: str) -> None:
        configuration = RobertaConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        self = RobertaForSequenceClassification.from_pretrained(model_name, config=configuration)

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, output_attentions = None, output_hidden_states = None, return_dict = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None and self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        # return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)