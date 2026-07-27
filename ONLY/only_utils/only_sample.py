import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
import torch.nn.functional as F



def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    model_kwargs_pos = model_kwargs.copy()
    model_kwargs_neg = model_kwargs.copy()
    
    print("use_ritual = ", model_kwargs.get("use_ritual"))
    print("use_vcd = ", model_kwargs.get("use_vcd"))
    print("use_m3id = ", model_kwargs.get("use_m3id"))
    print("use_only = ", model_kwargs.get("use_only"))
    
    
    t=0
    total_overlapping_index_len = []
    while True:
        ## For complementive & contrastive decoding
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_only = model_kwargs.get("use_only")

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        if use_only:
            outputs, logits_cd = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, # False
            output_hidden_states=output_hidden_states
        )
        else: 
            outputs, _ = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions, # True
                output_hidden_states=output_hidden_states
            )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        
        if use_ritual or use_vcd or use_m3id or use_only:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs["images_pos"] is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos, _ = self(
                    **model_inputs_pos,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :]

            elif model_kwargs["images_neg"] is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg, _ = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg, _ = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            
                
            ritual_alpha_pos = model_kwargs.get("ritual_alpha_pos") if model_kwargs.get("ritual_alpha_pos") is not None else 3
            ritual_alpha_neg = model_kwargs.get("ritual_alpha_neg") if model_kwargs.get("ritual_alpha_neg") is not None else 1
            ritual_beta = model_kwargs.get("ritual_beta") if model_kwargs.get("ritual_beta") is not None else 0.1
            js_gamma = model_kwargs.get("js_gamma") if model_kwargs.get("js_gamma") is not None else 0.1


            # set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(ritual_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            if use_ritual:
                diffs = next_token_logits + ritual_alpha_pos * next_token_logits_pos
            elif use_vcd:
                # diffs = next_token_logits + ritual_alpha_pos * next_token_logits_neg
                diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02*t))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg)*(1-gamma_t)/gamma_t
                t += 1
            elif use_only:
                assert logits_cd is not None
                debug_tvd_active = model_kwargs.get("debug_tvd", False)

                if isinstance(logits_cd, tuple):
                    # Proposal 4a / 4b: 3-Branch Symmetric Contrastive Decoding
                    logits_cd_vis, logits_cd_txt = logits_cd
                    next_token_logits_vis = logits_cd_vis[:, -1, :]
                    next_token_logits_txt = logits_cd_txt[:, -1, :]

                    p_normal = nn.functional.softmax(next_token_logits, dim=-1)
                    p_text = nn.functional.softmax(next_token_logits_txt, dim=-1)
                    p_visual = nn.functional.softmax(next_token_logits_vis, dim=-1)

                    # Cross-Modal Conflict Disagreement Scoring (TVD)
                    d_text = 0.5 * torch.sum(torch.abs(p_normal - p_text), dim=-1)
                    d_visual = 0.5 * torch.sum(torch.abs(p_normal - p_visual), dim=-1)

                    # Hyperparameters with None-safe fallbacks
                    gamma_1 = model_kwargs.get("gamma_1") if model_kwargs.get("gamma_1") is not None else js_gamma
                    gamma_2 = model_kwargs.get("gamma_2") if model_kwargs.get("gamma_2") is not None else js_gamma
                    alpha_1 = model_kwargs.get("alpha_1") if model_kwargs.get("alpha_1") is not None else ritual_alpha_pos
                    alpha_2 = model_kwargs.get("alpha_2") if model_kwargs.get("alpha_2") is not None else ritual_alpha_neg
                    alpha_3 = model_kwargs.get("alpha_3") if model_kwargs.get("alpha_3") is not None else ritual_alpha_pos
                    alpha_4 = model_kwargs.get("alpha_4") if model_kwargs.get("alpha_4") is not None else ritual_alpha_neg

                    # Dual-Branch Adaptive Anchored Decoding Rule
                    # Base anchor scaling factor (1 + alpha) prevents raw logit sign inversion when subtracting contrastive heads
                    anchor_scale = 1.0

                    if d_text < gamma_1:
                        delta_text = alpha_1 * next_token_logits_txt
                        regime_txt = "comp_txt"
                    else:
                        delta_text = -alpha_2 * next_token_logits_txt
                        anchor_scale += alpha_2
                        regime_txt = "contr_txt"

                    if d_visual < gamma_2:
                        delta_visual = alpha_3 * next_token_logits_vis
                        regime_vis = "comp_vis"
                    else:
                        delta_visual = -alpha_4 * next_token_logits_vis
                        anchor_scale += alpha_4
                        regime_vis = "penal_vis"

                    # 3-Branch Resolution Formula with Base Logit Anchoring
                    diffs = anchor_scale * next_token_logits + delta_text + delta_visual

                    if debug_tvd_active:
                        print(f"[TVDT t={len(total_overlapping_index_len)}] d_txt={d_text.item():.4f} ({regime_txt}) d_vis={d_visual.item():.4f} ({regime_vis})")
                        total_overlapping_index_len.append((d_text.item(), d_visual.item()))
                    else:
                        total_overlapping_index_len.append((d_text.item(), d_visual.item()))
                else:
                    # Single contrastive branch (Proposals 0-3)
                    next_token_logits_cd = logits_cd[:, -1, :]
                    tvd = 0.5 * torch.sum(torch.abs(nn.functional.softmax(next_token_logits, dim=-1) - nn.functional.softmax(next_token_logits_cd, dim=-1)))

                    if tvd < js_gamma:
                        diffs = next_token_logits + ritual_alpha_pos * next_token_logits_cd
                    else:
                        diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_cd

                    if debug_tvd_active:
                        regime = "comp" if tvd.item() < js_gamma else "contr"
                        print(f"[TVDT t={len(total_overlapping_index_len)}] tvd={tvd.item():.4f} regime={regime}")
                        total_overlapping_index_len.append((tvd.item(), regime == "contr"))
                    else:
                        total_overlapping_index_len.append(tvd.item())

            # logits = next_token_logits
            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## ritual_comments: apply temperature warping and top-k filtering in contrastive decoding
            logits = logits_processor(input_ids, logits)
            logits = logits_warper(input_ids, logits)

            next_token_scores = logits
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        ## ritual_comments: update model_kwargs_ritual for complementive & contrastive decoding
        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_vcd or use_m3id:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids, decoder_attentions
            # return SampleDecoderOnlyOutput(
            #     sequences=input_ids,
            #     scores=scores,
            #     attentions=decoder_attentions,
            #     hidden_states=decoder_hidden_states,
            # )
    else:
        return input_ids, total_overlapping_index_len

def evolve_only_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample