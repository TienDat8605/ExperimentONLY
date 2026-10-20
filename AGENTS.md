# ExperimentONLY — ADBD research workspace

This repository owns the ADBD implementation, experiment evidence, and ACOMPA26_P02 paper.
Read [CONTEXT.md](CONTEXT.md) for ownership and the method → evaluation → paper workflow.

| Task | Start here |
|---|---|
| Edit the current paper | [paper/ACOMPA26_P02/CONTEXT.md](paper/ACOMPA26_P02/CONTEXT.md) |
| Change masks, model integration, or decoding | [ONLY/CONTEXT.md](ONLY/CONTEXT.md) |
| Run or inspect evaluation | [ONLY/eval_bench/CONTEXT.md](ONLY/eval_bench/CONTEXT.md) |
| Find raw results or record an experiment | [logs/CONTEXT.md](logs/CONTEXT.md) |
| Configure assets or runtime | [data/CONTEXT.md](data/CONTEXT.md), [README.md](README.md) |
| Verify code changes | [tests/CONTEXT.md](tests/CONTEXT.md) |
| Read method notes or old writing drafts | [docs/method/CONTEXT.md](docs/method/CONTEXT.md) |
| Find earlier manuscripts | [paper/versions/CONTEXT.md](paper/versions/CONTEXT.md) |
| Inspect organization and migration proof | [docs/reorganization/CONTEXT.md](docs/reorganization/CONTEXT.md) |
| Start an experiment record | [_templates/experiment-record.md](_templates/experiment-record.md) |

## Persistent workspace instructions

At session start, read `/home/dat/.agents/skills/icm-architect/SKILL.md`.
Load only the contract, sources, and evidence needed for the current task.
The root `ACOMPA26_P02` symlink is a compatibility entry; edit its canonical paper target.
`/home/dat/HCMUS/experiment` is a separate comparison repository. Keep its proposal IDs distinct.

Prefer `colab ssh` for runtime execution and SSH-based file transfer; verify installed options first.
Use `colab ssh -s SESSION -i ~/.ssh/id_ed25519_colab` for interactive access.
For scripted OpenSSH, use `ProxyCommand="colab ssh --proxy-mode -s %h"` and `root@SESSION`.
`ssh posedflow-inference COMMAND` and transfers to `posedflow-inference:` use the configured proxy/key.
Scope `IdentityAgent none` and `IdentitiesOnly yes` to Colab hosts. Never print private keys.
Use the Colab CLI for discovery, creation, status, recovery, and SSH fallback.
Reuse named sessions, preserve running jobs, and never use `--rm` on persistent study sessions.
