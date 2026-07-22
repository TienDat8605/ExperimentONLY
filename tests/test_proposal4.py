import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proposal4 = load_module("proposal4_under_test", ROOT / "ONLY/only_utils/proposal4_utils.py")


class Proposal4ConsensusTests(unittest.TestCase):
    def test_zero_residual_falls_back_to_layer0_only(self):
        normal = torch.tensor([[0.1, 0.4, -0.2]])
        experts = normal[:, None, :].repeat(1, 4, 1)

        result, diagnostics = proposal4.consensus_logits(normal, experts, gamma=0.2)

        self.assertTrue(torch.allclose(result, normal + 3.0 * normal))
        self.assertEqual(diagnostics["activation_ratio"].item(), 0.0)

    def test_consensus_output_is_bounded_and_finite(self):
        normal = torch.tensor([[2.0, 0.0, -1.0]])
        experts = torch.tensor(
            [[[0.0, 2.0, -1.0], [0.1, 1.9, -1.0], [0.2, 1.8, -1.0], [0.3, 1.7, -1.0]]]
        )

        result, diagnostics = proposal4.consensus_logits(normal, experts, gamma=0.2)

        self.assertEqual(result.shape, normal.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue((diagnostics["step_gate"] >= 0).all())
        self.assertTrue((diagnostics["step_gate"] <= 1).all())


class Proposal4EntropyTests(unittest.TestCase):
    def test_uniform_attention_is_length_normalized(self):
        probabilities = torch.full((1, 4, 1, 10), 0.1)
        gate, score = proposal4.soft_head_gate(probabilities, 3, 8)

        self.assertTrue(torch.allclose(score, torch.zeros_like(score), atol=1e-5))
        self.assertTrue(torch.allclose(gate, torch.full_like(gate, 0.5), atol=1e-5))

    def test_invalid_image_span_is_rejected(self):
        probabilities = torch.full((1, 2, 1, 10), 0.1)
        with self.assertRaises(ValueError):
            proposal4.soft_head_gate(probabilities, 8, 12)


if __name__ == "__main__":
    unittest.main()
