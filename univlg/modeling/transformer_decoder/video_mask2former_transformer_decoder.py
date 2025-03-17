# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging

import ipdb
import torch
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from univlg.modeling.backproject.backproject import voxel_map_to_source
from univlg.modeling.meta_arch.language_encoder import LanguageEncoder
from univlg.modeling.meta_arch.self_cross_attention_layers import (
    MLP,
    CrossAttentionLayer,
    FFNLayer,
    SelfAttentionLayer,
)
from univlg.modeling.transformer_decoder.position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine3D
from univlg.utils.misc import nanmax, nanmin
from univlg.modeling.generation_head.t5_head import T5
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

st = ipdb.set_trace


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        decoder_3d: bool,
        cfg=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # self.num_frames = num_frames
        self.decoder_3d = decoder_3d
        self.hidden_dim = hidden_dim
        self.cfg = cfg

        self.pe_layer, self.pe_layer_2d = self.init_pe()

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_f_to_q = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.vis_output_cross_attn = nn.ModuleList()
        self.transformer_text_cross_attention_layers = nn.ModuleList()
        self.vis_output_ffn = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

            if self.cfg.VIS_LANG_ATTN:
                self.vis_output_cross_attn.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                        normalize_before=pre_norm,
                    )
                )

                self.vis_output_ffn.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                        normalize_before=pre_norm,
                    )
                )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # output FFNs
        if self.mask_classification and not self.cfg.MODEL.OPEN_VOCAB:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        if self.cfg.MODEL.OPEN_VOCAB:
            self.lang_encoder = LanguageEncoder(cfg, d_model=hidden_dim)
            self.max_seq_len = cfg.MODEL.MAX_SEQ_LEN

            self.lang_pos_embed = nn.Embedding(self.max_seq_len, hidden_dim)
            self.class_embed = nn.Linear(hidden_dim, hidden_dim)

        if self.cfg.GENERATION:
            if self.cfg.AR_LLM:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct" if self.cfg.AR_INSTRUCT else "HuggingFaceTB/SmolLM2-135M"
                device = "cuda"
                self.lang_generation_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                self.lang_generation_head = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
                self.lang_generation_head.input_proj = nn.Sequential(nn.Linear(hidden_dim, self.lang_generation_head.model.config.hidden_size), nn.LayerNorm(self.lang_generation_head.model.config.hidden_size))
                use_outer_layers_only = True
                if use_outer_layers_only:
                    num_layers = len(self.lang_generation_head.model.layers)
                    middle_start = num_layers // 6
                    middle_end = num_layers - middle_start
                    self.lang_generation_head.model.layers = self.lang_generation_head.model.layers[:middle_start] + self.lang_generation_head.model.layers[middle_end:]
            else:
                self.lang_generation_head = T5(input_size=hidden_dim)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES

        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["decoder_3d"] = cfg.MODEL.DECODER_3D
        ret["cfg"] = cfg

        return ret

    def init_pe(self):
        pe_layer_3d, pe_layer_2d = None, None
        N_steps = self.hidden_dim // 2
        if self.cfg.MODEL.DECODER_3D:
            pe_layer_3d = PositionEmbeddingLearned(
                dim=3, num_pos_feats=self.hidden_dim
            )
        if self.cfg.MODEL.DECODER_2D:
            pe_layer_2d = PositionEmbeddingSine3D(
                N_steps, normalize=True, add_temporal=self.cfg.ADD_TEMPORAL
            )
        return pe_layer_3d, pe_layer_2d

    def open_vocab_class_pred(self, decoder_output, text_feats):
        class_embed = self.class_embed(decoder_output)
        query_feats = F.normalize(class_embed, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        output_class = torch.einsum("bqc,sbc->bqs", query_feats / 0.07, text_feats)
        return output_class

    def init_object_queries(
        self, bs
    ):
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        return query_embed, output

    def forward(
        self,
        mask_features,
        shape=None,
        mask_features_xyz=None,
        mask_features_p2v=None,
        segments=None,
        decoder_3d=False,
        captions=None,
        actual_decoder_3d=False,
        scannet_all_masks_batched=None,
        max_valid_points=None,
        tokenized_answer=None,
        answers=None,
    ):
        """Language Conditioned Mask Decoder Head
        
        Keyword arguments:
            mask_features (features coming from the visual backbone):
                In 2D: B*V, C, H, W
                In 3D: B, C, N, 1
            shape: [B, V]
            mask_features_xyz: B, N, 3
            mask_features_p2v: B, N
            segments (optional) these are supervoxels, popularly used in ScanNet benchmarks: B, N
            decoder_3d: whether the decoder is 3D or not
            captions: [list of text sentences] of length B
            actual_decoder_3d: whether the data is really 3D or (lifted) 3D
            scannet_all_masks_batched (optional) GT setup of Referit3D assume GT seg masks: B, N
            max_valid_points: [list of integers] of length B (number of valid points in each batch; only relevant for training)
            tokenized_answer: [list of tokenized answers] of length B for Generation tasks
        Return: 
            outputs: dictionary containing the following main keys:
            pred_logits: predicted logits [B, num_queries, num_text_tokens+1] (distribution over text tokens for classification)
            pred_masks: predicted masks (distribution over pixels for segmentation) 
                        In 3D: [B, num_queries, N]
                        In 2D: [B*V, num_queries, H, W]
            generation_language: predicted language for Generation tasks [list of answers] of length B
                        
        """
        
        voxelize = (
            decoder_3d and self.cfg.INPUT.VOXELIZE
        )

        if shape is None:
            assert not decoder_3d
            shape = [mask_features.shape[0], 1]
        bs, v = shape
        pe_layer = self.pe_layer if decoder_3d else self.pe_layer_2d

        if not decoder_3d:
            bv, c_m, h_m, w_m = mask_features.shape
            mask_features = mask_features.view(bs, v, c_m, h_m, w_m)
            self.forward_prediction_heads = self.forward_prediction_heads2D
            if voxelize:
                mask_features = scatter_mean(
                    mask_features.permute(0, 1, 3, 4, 2).flatten(1, 3),
                    mask_features_p2v,
                    dim=1,
                )  # b, n, c
        else:
            if (
                (
                    self.cfg.HIGH_RES_INPUT
                    and not self.training
                    and not self.cfg.USE_GHOST_POINTS
                )
                or (
                    self.cfg.NO_POINTREND
                    and not (self.cfg.HIGH_RES_INPUT and self.cfg.HIGH_RES_SUBSAMPLE)
                )
            ) and self.cfg.INPUT.VOXELIZE:
                mask_features = (
                    mask_features.reshape(
                        bs, v, -1, mask_features.shape[-2], mask_features.shape[-1]
                    )
                    .permute(0, 1, 3, 4, 2)
                    .flatten(1, 3)
                )
                mask_features = scatter_mean(
                    mask_features,
                    mask_features_p2v,
                    dim=1,
                )[..., None].permute(
                    0, 2, 1, 3
                )  # b, c, n, 1

            assert mask_features.shape[-1] == 1, mask_features.shape
            mask_features = mask_features[..., 0]
            # voxelize (mask features are already voxelized)
            assert mask_features_xyz.shape[-2] == mask_features.shape[-1]
            if self.cfg.USE_GT_MASKS:
                mask_features = scatter_mean(
                    mask_features.permute(0, 2, 1), scannet_all_masks_batched.long(), dim=1
                ).permute(
                    0, 2, 1
                )
            elif self.cfg.USE_SEGMENTS:
                mask_features = scatter_mean(
                    mask_features.permute(0, 2, 1), segments, dim=1
                ).permute(
                    0, 2, 1
                )  # B, C, N

            self.forward_prediction_heads = self.forward_prediction_heads3D

        text_feats = None
        text_attn_mask = None
        if self.cfg.MODEL.OPEN_VOCAB:
            assert captions is not None
            text_feats, text_attn_mask = self.lang_encoder(
                captions
            )  # B X S X C
            text_feats = text_feats.permute(1, 0, 2)  # S X B X C

            # add these text features as text queries
            bs = text_feats.shape[1]
            lang_pos_embed = self.lang_pos_embed.weight[:, None].repeat(1, bs, 1)[
                : text_feats.shape[0]
            ]

        if decoder_3d:
            if self.cfg.USE_GT_MASKS:
                mask_features_xyz_segments = scatter_mean(
                    mask_features_xyz, scannet_all_masks_batched, dim=1
                )
            elif self.cfg.USE_SEGMENTS and segments is not None:
                mask_features_xyz_segments = scatter_mean(
                    mask_features_xyz, segments, dim=1
                )
            else:
                mask_features_xyz_segments = mask_features_xyz

        mask_features_pos = pe_layer(mask_features_xyz_segments).permute(1, 0, 2)

        query_embed, output = self.init_object_queries(bs)

        query_pad_mask = None
        predictions_class = []
        predictions_mask = []
        if self.cfg.GENERATION:
            predictions_generation = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            segments=segments,
            text_feats=text_feats,
            scannet_all_masks_batched=scannet_all_masks_batched,
        )
        if self.cfg.GENERATION:
            generation_features = output
            if self.cfg.DETACH_GENERATION_LOSS:
                generation_features = generation_features.detach()
            if self.cfg.AR_LLM:
                generation_logits, generation_labels = self.forward_generation_head(generation_features, captions, answers)
            else:
                generation_logits = self.lang_generation_head(generation_features.permute(1, 0, 2), None, tokenized_answer)
            predictions_generation.append(generation_logits)

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            if self.cfg.MODEL.OPEN_VOCAB:
                # attention to text tokens
                query_pad_mask = torch.cat(
                    [
                        torch.zeros(
                            output.shape[1], output.shape[0], device=output.device
                        ).bool(),
                        text_attn_mask,
                    ],
                    1,
                )
                output = torch.cat([output, text_feats], dim=0)
                if i == 0:
                    query_embed = torch.cat([query_embed, lang_pos_embed], dim=0)
                attn_mask = torch.cat(
                    [
                        attn_mask,
                        torch.zeros(
                            attn_mask.shape[0],
                            text_feats.shape[0],
                            attn_mask.shape[2],
                            device=output.device,
                        ).bool(),
                    ],
                    1,
                )
            
            src_attn = mask_features.permute(2, 0, 1)
            pos_attn = mask_features_pos

            output = self.transformer_cross_attention_layers[i](
                output,
                src_attn,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_attn,
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=query_pad_mask,
                query_pos=query_embed,
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # attention from mask_features to output
            if self.cfg.VIS_LANG_ATTN:
                mask_features = self.vis_output_cross_attn[i](
                    mask_features.permute(2, 0, 1),
                    output,
                    pos=query_embed,
                    query_pos=mask_features_pos,
                )
                mask_features = self.vis_output_ffn[i](mask_features)
                mask_features = mask_features.permute(1, 2, 0)

            if self.cfg.MODEL.OPEN_VOCAB:
                output, text_feats = (
                    output[: -text_feats.shape[0]],
                    output[-text_feats.shape[0] :],
                )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                segments=segments,
                text_feats=text_feats,
                scannet_all_masks_batched=scannet_all_masks_batched,
            )

            if self.cfg.GENERATION:
                generation_features = output
                if self.cfg.DETACH_GENERATION_LOSS:
                    generation_features = generation_features.detach()

                if self.cfg.AR_LLM:
                    generation_logits, generation_labels = self.forward_generation_head(generation_features, captions, answers)
                else:
                    generation_logits = self.lang_generation_head(generation_features.permute(1, 0, 2), None, tokenized_answer)
                predictions_generation.append(generation_logits)


            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if self.cfg.GENERATION and not self.training:
            if self.cfg.AR_LLM:
                generation_language = self.eval_generation_head(output, captions)
            else:
                generation_language = self.lang_generation_head(output.permute(1, 0, 2), None)
        else:
            generation_language = None

        assert len(predictions_class) == self.num_layers + 1

        predictions_boxes = None
        if self.cfg.USE_BOX_LOSS and actual_decoder_3d and self.training:
            predictions_boxes = []
            for i in range(len(predictions_mask)):
                masks = predictions_mask[i] > 0

                # remove padded tokens
                for j in range(len(max_valid_points)):
                    masks[j, :, max_valid_points[j]:] = False

                pc = mask_features_xyz_segments
                pc = pc[:, None].repeat(1, masks.shape[1], 1, 1)
                pc[torch.where(masks == 0)] = torch.nan

                boxes = torch.cat([
                    nanmin(pc, dim=2)[0], nanmax(pc, dim=2)[0]
                ], 2)

                # if only one point is in the mask, the box will still be too small
                boxes[masks.sum(2) <= 1] = torch.tensor([0, 0, 0, 1e-2, 1e-2, 1e-2], device=boxes.device)
                predictions_boxes.append(boxes)
        out = {
            "text_attn_mask": text_attn_mask,
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            'pred_boxes': predictions_boxes[-1] if predictions_boxes is not None else None,
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask, predictions_boxes,
                predictions_generation if self.cfg.GENERATION else None,
            ),
            "generation_logits": predictions_generation[-1] if self.cfg.GENERATION else None,
            'generation_language': generation_language if self.cfg.GENERATION else None,
        }

        if self.cfg.AR_LLM:
            out['generation_labels'] = generation_labels

        return out
    
    def forward_generation_head(self, generation_features, captions, answers):
        if answers is None:
            assert not self.training
            answers = ["debug"] * len(captions)
        device = generation_features.device
        assert len(captions) == generation_features.shape[1]
        map_feats = self.lang_generation_head.input_proj(generation_features.permute(1, 0, 2))
        inputs_embeds = []
        max_tokens = 256 # To be safe. We can probably get away with 200 or even less.
        labels = torch.full((generation_features.shape[1], max_tokens), -100, dtype=torch.long).to(device)
        for k in range(generation_features.shape[1]):
            if self.cfg.AR_EMBED:
                if self.cfg.AR_INSTRUCT:
                    user_messages = [{"role": "user", "content": captions[k]}]
                    input_text = self.lang_generation_tokenizer.apply_chat_template(user_messages, tokenize=False)
                    _inputs = self.lang_generation_tokenizer.encode(input_text, return_tensors="pt").to(device)
                    _chat_template = '''{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}'''
                    _answer_messages = [{"role": "assistant", "content": answers[k]}]
                    _answer_text = self.lang_generation_tokenizer.apply_chat_template(_answer_messages, chat_template=_chat_template, tokenize=False)
                    _answers = self.lang_generation_tokenizer.encode(_answer_text, return_tensors="pt").to(device)
                else:
                    _inputs = self.lang_generation_tokenizer.encode(self.lang_generation_tokenizer.bos_token + captions[k] + self.lang_generation_tokenizer.eos_token, return_tensors="pt").to(device)
                    _answers = self.lang_generation_tokenizer.encode(answers[k] + self.lang_generation_tokenizer.eos_token, return_tensors="pt").to(device)
                
                _inputs_embs = self.lang_generation_head.model.embed_tokens(_inputs)
                _answers_embs = self.lang_generation_head.model.embed_tokens(_answers)
                _inputs_embeds = torch.cat([map_feats[[k]], _inputs_embs, _answers_embs], dim=1)

                start_pos = _inputs_embeds.shape[1] - _answers.shape[1]
                labels[k, start_pos:start_pos+_answers.shape[1]] = _answers
                if self.cfg.AR_INSTRUCT:
                    labels[k, start_pos+_answers.shape[1]:] = self.lang_generation_tokenizer.pad_token_id
                else:
                    labels[k, :start_pos] = _inputs
                    labels[k, start_pos+_answers.shape[1]:] = self.lang_generation_tokenizer.eos_token_id
                    if labels[k].min().item() < 0:
                        breakpoint()
                _inputs_embeds = torch.nn.functional.pad(_inputs_embeds, (0, 0, 0, max_tokens - _inputs_embeds.shape[1]), value=0)
                inputs_embeds.append(_inputs_embeds)
            else:
                user_messages = [{"role": "user", "content": captions[k]}, {"role": "assistant", "content": answers[k]}]
                _input_text = self.lang_generation_tokenizer.apply_chat_template(user_messages, tokenize=False)
                __inputs = self.lang_generation_tokenizer.encode(_input_text, return_tensors="pt", max_length=max_tokens, padding='max_length').to(device)
                inputs_embeds.append(__inputs)

        inputs_embeds = torch.cat(inputs_embeds, dim=0)

        if self.cfg.AR_EMBED:
            generation_output = self.lang_generation_head(input_ids=None, inputs_embeds=inputs_embeds, labels=labels)
        else:
            generation_output = self.lang_generation_head(input_ids=inputs_embeds, labels=inputs_embeds)
            labels = inputs_embeds

        generation_logits = generation_output.logits.permute(0, 2, 1).float()
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        labels = labels[:, 1:].contiguous()
        # print(f"GT LLM loss: {generation_output.loss}")
        return generation_logits, labels
    
    def eval_generation_head(self, generation_features, captions):
        device = generation_features.device
        assert len(captions) == generation_features.shape[1]
        map_feats = self.lang_generation_head.input_proj(generation_features.permute(1, 0, 2))
        decoded = []
        for k in range(generation_features.shape[1]):
            if self.cfg.AR_EMBED:
                if self.cfg.AR_INSTRUCT:
                    messages = [{"role": "user", "content": captions[k]}]
                    input_text = self.lang_generation_tokenizer.apply_chat_template(messages, tokenize=False)
                    _inputs = self.lang_generation_tokenizer.encode(input_text, return_tensors="pt").to(device)
                    _inputs_embs = self.lang_generation_head.model.embed_tokens(_inputs)
                    
                    _chat_template = '''{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}'''
                    _answer_messages = [{"role": "assistant", "content": ""}]
                    _answer_text = self.lang_generation_tokenizer.apply_chat_template(_answer_messages, chat_template=_chat_template, tokenize=False)
                    _answers = self.lang_generation_tokenizer.encode(_answer_text, return_tensors="pt").to(device)
                    _answers_embs = self.lang_generation_head.model.embed_tokens(_answers)
                    _inputs_embeds = torch.cat([map_feats[[k]], _inputs_embs, _answers_embs], dim=1)
                else:
                    _inputs = self.lang_generation_tokenizer.encode(self.lang_generation_tokenizer.bos_token + captions[k] + self.lang_generation_tokenizer.eos_token, return_tensors="pt").to(device)
                    _inputs_embeds = self.lang_generation_head.model.embed_tokens(_inputs)

                if self.cfg.AR_INSTRUCT:
                    from transformers.generation import SuppressTokensLogitsProcessor, LogitsProcessorList
                    logits_processor = LogitsProcessorList([SuppressTokensLogitsProcessor([self.lang_generation_tokenizer.bos_token_id], device=device)])
                    generation_output = self.lang_generation_head.generate(input_ids=None, inputs_embeds=_inputs_embeds, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True, logits_processor=logits_processor)
                    decoded_ = self.lang_generation_tokenizer.batch_decode(generation_output, skip_special_tokens=True)
                    print(f"Question: {self.lang_generation_tokenizer.batch_decode(_inputs)}, Answer Prefix: {self.lang_generation_tokenizer.batch_decode(_answers)}, Answer: {self.lang_generation_tokenizer.batch_decode(generation_output, skip_special_tokens=True)}")
                else:
                    # generation_output = self.lang_generation_head.generate(input_ids=None, inputs_embeds=_inputs_embeds, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True, logits_processor=None)
                    # print(f"Question: {self.lang_generation_tokenizer.batch_decode(_inputs)}, Answer: {self.lang_generation_tokenizer.batch_decode(generation_output, skip_special_tokens=False)}")

                    generation_output = self.lang_generation_head.generate(input_ids=_inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True, logits_processor=None, pad_token_id=self.lang_generation_tokenizer.eos_token_id)
                    print(f"Answer: {self.lang_generation_tokenizer.batch_decode(generation_output, skip_special_tokens=False)}")

                    decoded_ = self.lang_generation_tokenizer.batch_decode(generation_output, skip_special_tokens=False)
                    decoded_ = [i.split('<|endoftext|>')[-2] for i in decoded_]
            else:
                user_messages = [{"role": "user", "content": captions[k]}]
                _input_text = self.lang_generation_tokenizer.apply_chat_template(user_messages, tokenize=False)
                __inputs = self.lang_generation_tokenizer.encode(_input_text, return_tensors="pt").to(device)
                
                generation_output = self.lang_generation_head.generate(input_ids=__inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True, logits_processor=None)
                decoded_ = self.lang_generation_tokenizer.batch_decode(generation_output, skip_special_tokens=True)
                print(f"Question: {self.lang_generation_tokenizer.batch_decode(__inputs)}, Answer: {self.lang_generation_tokenizer.batch_decode(generation_output, skip_special_tokens=False)}")

            assert len(decoded_) == 1
            decoded.append(decoded_[0])
        decoded = [i.removesuffix('\n').removeprefix('\n').split('\n')[-1] for i in decoded]
        return decoded

    def forward_prediction_heads3D(
        self,
        output,
        mask_features,
        segments=None,
        text_feats=None,
        scannet_all_masks_batched=None,
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        outputs_class = self.open_vocab_class_pred(
            decoder_output, text_feats,
        )

        mask_embed = self.mask_embed(decoder_output)

        segment_mask = torch.einsum("bqc,bcn->bqn", mask_embed, mask_features)
        if self.cfg.USE_GT_MASKS:
            output_mask = voxel_map_to_source(
                segment_mask.permute(0, 2, 1), scannet_all_masks_batched
            ).permute(0, 2, 1)
        elif self.cfg.USE_SEGMENTS:
            output_mask = voxel_map_to_source(
                segment_mask.permute(0, 2, 1), segments
            ).permute(0, 2, 1)
        else:
            output_mask = segment_mask
            segment_mask = None

        attn_mask = segment_mask if self.cfg.USE_SEGMENTS else output_mask
        
        attn_mask = (
            attn_mask.sigmoid()
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        if self.cfg.USE_SEGMENTS:
            output_mask = segment_mask

        return outputs_class, output_mask, attn_mask
    
    
    def forward_prediction_heads2D(
        self,
        output,
        mask_features,
        segments=None,
        text_feats=None,
        scannet_all_masks_batched=None
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        outputs_class = self.open_vocab_class_pred(
            decoder_output, text_feats,
        )

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)

        attn_mask = outputs_mask.flatten(2)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boxes, outputs_lang):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_boxes is not None and self.cfg.GENERATION:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "generation_logits": d}
                for a, b, c, d in zip(
                    outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1], outputs_lang[:-1]
                )
            ]
        elif outputs_boxes is not None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1])
            ]
        elif self.mask_classification:
            if self.cfg.GENERATION:
                return [
                    {"pred_logits": a, "pred_masks": b, "generation_logits": c}
                    for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_lang[:-1])
                ]
            else:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b, in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
