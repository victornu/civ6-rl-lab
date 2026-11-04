# civ6-rl-lab

Deep reinforcement learning for Civilization VI through the civ6-host client.

This repository is a source snapshot of the PyTorch learner and its monitoring
tools. The model keeps the original `civ6rl.learning` import paths. The legacy
native campaign launchers, optional Windows worker containment, machine-specific
setup and checkpoints still depend on the original local lab workspace.

## Included sources

- Existing map CNN, entity representations and per-player/per-episode LSTM.
- Human-availability action masking, recurrent PPO, GAE/bootstrap and full checkpoint resume.
- Player-only baseline and separate asymmetric critic; privileged training sidecars never enter actor inputs, masks, memory or deployment artifacts.
- Collection, curriculum, held-out evaluation, checkpoints and learning monitor.
- Learner and worker transports, plus the Phase 1 training plan.

The host API and native runtime are maintained in the separate
[`civ6-host` repository](https://github.com/victornu/civ6-host). The snapshot
retains two pure data validators used by the learner's legacy wire formats.
The integration of the learner with the extracted host remains in development.

To install the source and run the included tests:

```shell
python -m pip install -e .
python -m unittest discover -s tests
```

Game binaries, official captures and databases, private traces, credentials and
model weights are stored separately from this source snapshot.
