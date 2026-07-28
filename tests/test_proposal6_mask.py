import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ONLY"))

from only_utils.proposal6_mask import (  # noqa: E402
    continuous_tver_evidence,
    proposal6_keep_masks,
    resolve_proposal6_layers,
    update_proposal6_mask,
    validate_alpha_bounds,
)


class Proposal6MaskTest(unittest.TestCase):
    def test_default_layers_accumulate_everywhere_except_final_layer(self):
        self.assertEqual(resolve_proposal6_layers(32, None), tuple(range(31)))

    def test_sparse_layers_are_exact_and_sorted(self):
        self.assertEqual(resolve_proposal6_layers(32, [5, 0, 2]), (0, 2, 5))

    def test_invalid_layer_selections_are_rejected(self):
        for layers in ([], [0, 0], [0, 31], [-1, 0]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                resolve_proposal6_layers(32, layers)

    def test_first_layer_initializes_continuous_complementary_state(self):
        ratio = torch.tensor([0.5, 1.0, 2.0, 4.0])
        state, stats = update_proposal6_mask(ratio, None)

        torch.testing.assert_close(state[0], continuous_tver_evidence(ratio))
        torch.testing.assert_close(state[1], 1.0 - state[0])
        self.assertTrue(stats["initialized"].item())
        self.assertTrue(torch.equal(stats["gain"], torch.zeros_like(ratio)))

    def test_masks_are_disjoint_and_exhaustive(self):
        state = torch.tensor([[0.2, 0.5, 0.8], [0.8, 0.5, 0.2]])
        visual_keep, text_keep = proposal6_keep_masks(state)

        self.assertFalse(torch.any(visual_keep & text_keep).item())
        self.assertTrue(torch.all(visual_keep | text_keep).item())
        self.assertTrue(torch.equal(text_keep, ~visual_keep))

    def test_confident_innovation_receives_larger_per_head_gain(self):
        ratio = torch.tensor([-4.0, 0.0, 4.0])
        previous_visual = torch.tensor([0.9, 0.5, 0.1])
        previous = torch.stack((previous_visual, 1.0 - previous_visual))

        _, stats = update_proposal6_mask(ratio, previous, alpha_min=0.05, alpha_max=0.50)
        gain = stats["gain"]

        self.assertGreater(gain[0].item(), gain[1].item())
        self.assertGreater(gain[2].item(), gain[1].item())
        self.assertGreaterEqual(gain.min().item(), 0.05)
        self.assertLessEqual(gain.max().item(), 0.50)

    def test_update_preserves_complementary_state(self):
        first, _ = update_proposal6_mask(torch.tensor([0.2, 0.7, 1.5]), None)
        second, _ = update_proposal6_mask(torch.tensor([1.5, 0.7, 0.2]), first)
        torch.testing.assert_close(second[1], 1.0 - second[0])

    def test_invalid_alpha_bounds_are_rejected(self):
        for bounds in ((-0.1, 0.5), (0.6, 0.5), (0.1, 1.1)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_alpha_bounds(*bounds)


if __name__ == "__main__":
    unittest.main()
