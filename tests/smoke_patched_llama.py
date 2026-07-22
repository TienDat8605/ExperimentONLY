"""CPU smoke test for the runtime-patched Transformers 4.31 LLaMA module."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "ONLY/patches/modeling_llama.py"


def load_patched_module():
    name = "transformers.models.llama.modeling_llama_proposal4_smoke"
    spec = importlib.util.spec_from_file_location(name, PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    patched = load_patched_module()
    config = patched.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        use_cache=True,
    )
    model = patched.LlamaModel(config).eval()
    expert_layers = (0, 8, 16, 24)

    with torch.no_grad():
        first = model(
            inputs_embeds=torch.randn(1, 620, config.hidden_size),
            attention_mask=torch.ones(1, 620, dtype=torch.long),
            use_cache=True,
            use_only=True,
            proposal=4,
            expert_layers=expert_layers,
            image_token_start=35,
            image_token_end=611,
        )
        normal, experts = first[0].last_hidden_state, first[1]
        assert normal.shape == (1, 620, config.hidden_size)
        assert experts.shape == (5, 1, 620, config.hidden_size)

        second = model(
            inputs_embeds=torch.randn(1, 1, config.hidden_size),
            attention_mask=torch.ones(1, 621, dtype=torch.long),
            past_key_values=first[0].past_key_values,
            use_cache=True,
            use_only=True,
            proposal=4,
            expert_layers=expert_layers,
            image_token_start=35,
            image_token_end=611,
        )
        assert second[0].last_hidden_state.shape == (1, 1, config.hidden_size)
        assert second[1].shape == (5, 1, 1, config.hidden_size)

    print("proposal-4 patched LLaMA forward + KV cache: OK")


if __name__ == "__main__":
    main()
