---
name: Bug report
about: A command failed, crashed, or printed something wrong
labels: bug
---

**What you ran**

```bash
# the exact command(s), e.g. lfa train --workspace runs/my_domain --corpus data/my_domain
```

**What happened**

```text
# the tail of the output (the last ~40 lines), or the traceback
```

**What you expected**

**Environment**

- lfa-anchoring version or commit (`lfa --version`, or `git log --oneline -1` in the checkout):
- installed with `-c constraints-tested.txt`? (yes / no):
- Python (`python --version`):
- torch and CUDA (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`):
- GPU(s) and memory (`nvidia-smi --query-gpu=name,memory.total --format=csv`):
- model (e.g. Qwen/Qwen3-0.6B) and recipe, if you changed `--lambda`, `--mu` or the recipe:

**The workspace, if a run is involved**

The run's `config.json` and `training_history.json` (in `<workspace>/runs/stage1/`, or the
stage that failed) and the end of what `lfa train` printed help most. Leave out anything from
your corpus you do not want public.
