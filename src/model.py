from dataclasses import dataclass, field
import typing

import torch
from transformers import AutoModel, PreTrainedModel


class TextClassificationModel(torch.nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        hidden_dim: int,
        output_dim: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.headder = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: typing.Optional[torch.FloatTensor] = None,
        token_type_ids: typing.Optional[torch.LongTensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        head_mask: typing.Optional[torch.FloatTensor] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        output_attentions: typing.Optional[torch.FloatTensor] = None,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)

        logits = self.headder(pooled_output)

        return logits


# @torch.jit.script
# def mean_pooling(last_hidden_state, attention_mask):
#     attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
#     sum_hidden_state = torch.sum(last_hidden_state * attention_mask, 1)
#     sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
#     embeddings = sum_hidden_state / sum_mask
#     return embeddings
