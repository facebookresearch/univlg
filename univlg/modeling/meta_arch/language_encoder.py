# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    RobertaModel,
    RobertaTokenizerFast,
    AutoModel
)

st = ipdb.set_trace


class LanguageEncoder(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        if self.cfg.TEXT_ENCODER_TYPE == "clip":
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        elif self.cfg.TEXT_ENCODER_TYPE == "jina":
            self.text_encoder = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
            self.text_encoder.text_model.output_tokens = True
            self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
        else:
            t_type = "roberta-base"
            self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
            self.text_encoder = RobertaModel.from_pretrained(t_type)

        if cfg.MODEL.LANG_FREEZE_BACKBONE:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        _hidden_size = 768 if self.cfg.TEXT_ENCODER_TYPE == "jina" else self.text_encoder.config.hidden_size

        self.text_projector = nn.Sequential(
            nn.Linear(_hidden_size, d_model),
        )

    def forward(self, text):
        tokenized = self.tokenizer.batch_encode_plus(
            text,
            padding="longest" if not self.cfg.NON_PARAM_SOFTMAX else "max_length",
            return_tensors="pt",
            max_length=self.cfg.MODEL.MAX_SEQ_LEN
            if not self.cfg.TEXT_ENCODER_TYPE == "clip"
            else None,
            truncation=True,
        ).to(self.device)

        if self.cfg.TEXT_ENCODER_TYPE == "jina":
            encoded_text = self.text_encoder.text_model(tokenized['input_ids'])[-1] # B, N, D
            return self.text_projector(encoded_text), tokenized['attention_mask'].ne(1).bool()
        else:
            encoded_text = self.text_encoder(**tokenized)

        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        text_feats = encoded_text.last_hidden_state
        text_feats = self.text_projector(text_feats)

        return text_feats, text_attention_mask
