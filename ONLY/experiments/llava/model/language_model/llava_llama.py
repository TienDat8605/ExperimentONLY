#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append(".") # Adds higher directory to python modules path.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ...constants import IMAGE_TOKEN_INDEX


def _parse_expert_layers(value):
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, int):
        value = [value]
    layers = tuple(int(layer) for layer in value)
    if not layers:
        raise ValueError("expert_layers must contain at least one layer")
    if len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError(f"Invalid expert_layers: {layers}")
    return layers


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_pos: Optional[torch.FloatTensor] = None,
        images_neg: Optional[torch.FloatTensor] = None,
        use_ritual: Optional[bool] = None,
        use_vcd: Optional[bool] = None,
        use_m3id: Optional[bool] = None,
        use_only: Optional[bool] = None,
        enhance_layer_index: Optional[int] = 0,
        mask_alpha: Optional[float] = 0.2,
        debug_tvd: Optional[bool] = False,
        proposal: Optional[int] = 1,
        score_threshold: Optional[float] = 0.0,
        score_temperature: Optional[float] = 1.0,
        lambda_decay: Optional[float] = 0.3,
        expert_layers=(0, 8, 16, 24),
        entropy_temperature: Optional[float] = 1.0,
        consensus_min: Optional[float] = 0.75,
        consensus_strength: Optional[float] = 1.0,
        image_token_start: Optional[int] = None,
        image_token_end: Optional[int] = None,
        ritual_alpha_pos: Optional[torch.FloatTensor] = None,
        ritual_alpha_neg: Optional[torch.FloatTensor] = None,
        ritual_beta: Optional[torch.FloatTensor] = None,
        js_gamma: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        tokenizer=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if not use_only:
            outputs, _, _, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                debug_tvd=debug_tvd,
            )
        else:
            outputs, hidden_states_cd, cumulative_mask, score_accum = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_only=use_only,
                enhance_layer_index=enhance_layer_index,
                mask_alpha=mask_alpha,
                debug_tvd=debug_tvd,
                proposal=proposal,
                score_threshold=score_threshold,
                score_temperature=score_temperature,
                lambda_decay=lambda_decay,
                expert_layers=_parse_expert_layers(expert_layers),
                entropy_temperature=entropy_temperature,
                image_token_start=image_token_start,
                image_token_end=image_token_end,
            )
            if proposal == 4:
                # [experts, batch, sequence, hidden] -> [batch, experts, sequence, vocab]
                expert_hidden = hidden_states_cd + 0.5 * outputs[0].unsqueeze(0)
                logits_cd = self.lm_head(expert_hidden).permute(1, 0, 2, 3).contiguous()
            else:
                hidden_states_cd = hidden_states_cd + 0.5 * outputs[0]
                logits_cd = self.lm_head(hidden_states_cd)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), logits_cd if use_only else None
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        # The image placeholder is still present in GenerationMixin's growing
        # input_ids even after KV caching reduces the model input to one token.
        # Resolve the visual span here instead of relying on prompt-specific
        # constants such as 35:611.
        image_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        image_token_start = kwargs.get("image_token_start")
        image_token_end = kwargs.get("image_token_end")
        if image_positions.numel() > 0:
            starts = image_positions[:, 1].unique()
            if starts.numel() != 1:
                raise ValueError("proposal 4 currently requires a uniform image-token position per batch")
            image_token_start = int(starts.item())
            vision_tower = self.get_vision_tower()
            num_patches = int(getattr(vision_tower, "num_patches", 576))
            image_token_end = image_token_start + num_patches

        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "use_only": kwargs.get("use_only", None),
                "enhance_layer_index": kwargs.get("enhance_layer_index", None),
                "mask_alpha": kwargs.get("mask_alpha", 0.2),
                "debug_tvd": kwargs.get("debug_tvd", False),
                "proposal": kwargs.get("proposal", 1),
                "score_threshold": kwargs.get("score_threshold", 0.0),
                "score_temperature": kwargs.get("score_temperature", 1.0),
                "lambda_decay": kwargs.get("lambda_decay", 0.3),
                "expert_layers": kwargs.get("expert_layers", (0, 8, 16, 24)),
                "entropy_temperature": kwargs.get("entropy_temperature", 1.0),
                "image_token_start": image_token_start,
                "image_token_end": image_token_end,
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_pos(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_pos", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_neg(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_neg", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_m3id(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids[input_ids != -200].reshape(input_ids.shape[0], -1)}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask[:, :-1],
                "images": None,
            }
        )
        return model_inputs
    
    
AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
