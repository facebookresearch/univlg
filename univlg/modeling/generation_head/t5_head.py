# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

import ipdb
st = ipdb.set_trace

class T5(nn.Module):
    def __init__(self, variant='t5-small', input_size=768, use_projection=True, max_new_tokens=50):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(variant)
        self.tokenizer = AutoTokenizer.from_pretrained(variant)
        self.model.config.max_new_tokens = max_new_tokens
        hidden_size = self.model.config.d_model
        self.use_projection = use_projection
        if use_projection:
            self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size))
        else:
            assert input_size == hidden_size, "input_feat_size should be equal to hidden_size!"

    def forward(self, query_embeds, attention_masks, labels=None):
        if self.use_projection:
            query_embeds = self.input_proj(query_embeds)

        if labels is not None:
            outputs = self.model(encoder_outputs=[query_embeds], attention_mask=attention_masks, labels=labels)
            outputs = torch.clamp(outputs.logits, min=-100)
        else:
            outputs = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=query_embeds), attention_mask=attention_masks, do_sample=False)
            outputs = self.tokenizer.batch_decode(outputs[:, 1:], skip_special_tokens=True) # remove the decoder start token for T5 generation output.
        return outputs
