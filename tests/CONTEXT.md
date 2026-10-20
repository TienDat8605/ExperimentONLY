# Verification contract

One job: check affected method or command-line behavior without launching a full GPU evaluation.

Inputs: changed modules and the corresponding existing test file.

From the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_proposal4_local_cli.py'
python3 -m unittest discover -s tests -p 'test_proposal6_mask.py'
```

The CLI tests exercise parsing and validation with a deliberately missing environment.
The mask tests require PyTorch and check layer selection, bounded gains, and complementary masks.
They do not establish model quality, runtime-patch compatibility, or benchmark completeness.

Outputs: command exit status and test report. Human check: confirm the checks exercise the change;
record unavailable dependencies rather than installing or launching an evaluation implicitly.
For organizational integrity use `python3 docs/reorganization/verify.py`.
