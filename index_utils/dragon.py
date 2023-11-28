# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import transformers
from transformers import BertModel

from index_utils import utils


class Dragon(BertModel):

    def __init__(self, config, pooling="cls", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output['last_hidden_state']
        # last_hidden = last_hidden.masked_fill(
        #     ~attention_mask[..., None].bool(), 0.)

        if self.config.pooling == "average":
            raise ValueError("cls pooling should be used for DRAGON")
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        return emb


def load_retriever(model_path):
    cfg = utils.load_hf(transformers.AutoConfig, model_path)
    tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
    retriever = utils.load_hf(Dragon, model_path)
    return retriever, tokenizer
