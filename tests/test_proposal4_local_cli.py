import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_proposal4_local.sh"


class Proposal4LocalCliTest(unittest.TestCase):
    def run_cli(self, *arguments, trace=False):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["PROPOSAL4_VENV"] = str(Path(temp_dir) / "missing-venv")
            command = ["bash"]
            if trace:
                command.append("-x")
            command.extend([str(RUNNER), "run", *arguments])
            return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)

    def test_spaced_proposal6_options_are_accepted(self):
        result = self.run_cli(
            "--proposal", "6",
            "--layer", "0,2,5",
            "--mask-alpha-min", "0.1",
            "--mask-alpha-max", "0.4",
            trace=True,
        )
        output = result.stdout + result.stderr
        self.assertIn("PROPOSAL=6", output)
        self.assertIn("MASK_LAYERS=0,2,5", output)
        self.assertIn("MASK_ALPHA_MIN=0.1", output)
        self.assertIn("MASK_ALPHA_MAX=0.4", output)
        self.assertIn("Virtual environment not found", output)

    def test_equals_proposal6_options_are_accepted(self):
        result = self.run_cli(
            "--proposal=6",
            "--layer=0,1,2,3",
            "--mask-alpha-min=0.05",
            "--mask-alpha-max=0.50",
            trace=True,
        )
        output = result.stdout + result.stderr
        self.assertIn("PROPOSAL=6", output)
        self.assertIn("MASK_LAYERS=0,1,2,3", output)
        self.assertIn("Virtual environment not found", output)

    def test_duplicate_layers_are_rejected(self):
        result = self.run_cli("--proposal=6", "--layer=0,1,1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be unique", result.stderr)

    def test_final_layer_is_rejected(self):
        result = self.run_cli("--proposal=6", "--layer=0,31")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("layer 31 is reserved", result.stderr)

    def test_proposal6_options_are_rejected_for_other_proposals(self):
        result = self.run_cli("--proposal=4", "--layer=0,1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only valid with --proposal=6", result.stderr)

    def test_invalid_alpha_order_is_rejected(self):
        result = self.run_cli(
            "--proposal=6", "--mask-alpha-min=0.7", "--mask-alpha-max=0.2"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must satisfy 0 <= min <= max <= 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
