# Infrastructure: Training on a Restrictive Cloud Environment

This document exists because several files in this repo only make sense
in light of constraints that aren't visible from the code alone. It's
the technical companion to the dissertation's Analysis and Design and
Implementation chapters, kept here so a reader exploring the repo
directly (not just the write-up) has the same context.

## The core constraint

Training ran on a university-provided AWS "Learner Lab" account with
three hard limits discovered during development, not anticipated in
advance:

1. Only the `t3.large` instance type (8GB RAM, no GPU) is authorized —
   confirmed by a rejected launch attempt at `t3.xlarge`.
2. The account's IAM policy has an explicit **deny** on
   `ec2:StartInstances` — a stopped instance cannot be restarted; it
   must be terminated and relaunched fresh.
3. Instances are force-rebooted automatically on a roughly 4-hour cycle.

Any one of these would be manageable alone. Together, they mean a
training run longer than ~4 hours **will** be interrupted, repeatedly,
by something outside the application's control.

## What actually broke, and the fix, in the order it happened

1. **Full-scale training (32M interactions) exhausted memory.** Fixed
   by training on a documented 50,000-user subsample instead, plus
   switching RecBole's evaluation from exhaustive full-catalogue
   ranking to the standard `uni100` sampled protocol.

2. **RecBole's default checkpointing only saves on validation
   improvement**, not every epoch. Combined with the reboot cycle, this
   caused a real, unrecoverable loss of ~9 completed epochs of NCF
   training in one incident. Fixed by rewriting `train.py` to take
   manual control of the epoch loop (rather than calling RecBole's
   `run_recbole()` shortcut) and save an **unconditional** checkpoint
   after every single epoch, with a separate best-only checkpoint kept
   for genuine result reporting.

3. **Training didn't auto-resume after a reboot**, even with a cron
   `@reboot` entry configured. Root cause: cron runs commands with `sh`,
   not `bash`, so the bash-only `source` command failed silently; and
   cron's minimal `PATH` didn't include `tmux`'s location. Fixed by
   making the cron entry fully explicit:
   ```
   @reboot sleep 30 && /usr/bin/tmux new-session -d -s training '/bin/bash /home/ubuntu/fairness-auditor/run_training.sh >> /home/ubuntu/fairness-auditor/train_log.txt 2>&1'
   ```

4. **8GB of swap silently didn't survive a reboot** the first time it
   was set up, because it wasn't added to `/etc/fstab`. Fixed by adding:
   ```
   /swapfile none swap sw 0 0
   ```
   to `/etc/fstab` (and again for a second 16GB swapfile added later
   under real memory pressure from SASRec).

5. **LightGCN's evaluation cost turned out to be roughly fixed per user**,
   regardless of batch size — a full 50,000-user evaluation projected
   over 12 hours even after tuning, which cannot survive one reboot
   cycle and has no way to resume partway through. A custom resumable
   evaluation script was attempted and abandoned after it hit an
   internal RecBole error that would have required guessing at
   undocumented internals to fix — judged too risky for a number that
   goes into the dissertation's results. Instead, evaluation was run
   **locally**, on a machine with no forced reboot, using RecBole's own
   unmodified `evaluate()` — it completed after 16 hours 24 minutes,
   fully uninterrupted.

6. **SASRec needed two further, architecture-specific fixes**: its
   default loss function (`CE`) is incompatible with the project's
   negative-sampling configuration (fixed by switching to `loss_type:
   BPR`), and its Transformer attention cost scales with sequence
   length, so `MAX_ITEM_LIST_LENGTH` was reduced from 50 to 20 and batch
   size increased, cutting estimated total training time from
   impractical to roughly 70 hours — survivable given the per-epoch
   checkpointing from fix #2 above.

## Environment version-compatibility patches

RecBole 1.2.1 predates several dependency changes it wasn't built to
expect. All patched at the top of `train.py` (and, separately, in the
local evaluation script, since the local machine's package versions
differ from the cloud instance's):

- NumPy 2.0 removed `np.float_`, `np.complex_`, `np.unicode_` — restored
  as aliases before RecBole imports.
- RecBole imports `ray.tune` at load time for a hyperparameter-tuning
  feature that's never actually used here, and no `ray` wheel exists for
  some Python versions — satisfied with an empty stub module instead of
  installing it.
- PyTorch 2.6+ changed `torch.load`'s default to `weights_only=True`,
  which breaks loading RecBole's own checkpoint format — patched via a
  wrapped `torch.load` that restores the old default (safe here, since
  every checkpoint loaded was trained by this project itself).
- A local scipy version removed `dok_matrix._update` (used internally by
  LightGCN's graph construction) — restored as an alias to the current
  `dok_matrix.update`.

## Practical routine used throughout

Check on a running job with `tail -N train_log.txt` from a fresh
terminal connection, not by watching a long-held tmux pane directly —
the browser-based terminal used for this project repeatedly showed
stale or garbled output in panes left open a while, even when the
underlying process was healthy. `ps aux | grep "python train.py"` and
`free -h` were the reliable ground truth whenever the display looked
wrong.
