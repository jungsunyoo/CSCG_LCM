
# CoDA MiniGrid Project (updated)

This repo trains PPO on MiniGrid tasks **with** and **without** CoDA-based online
representation learning (split/merge). The CoDA algorithm here matches the split/merge
rules we used in our analysis:

- **Split**: uncertainty-aware threshold on *prospective contingency* with evidence gates
  (`n_threshold`, `min_presence_episodes`, `min_effective_exposure`), using a Beta posterior
  or Wilson lower bound for confidence.
- **Merge**: when `PC×RC` drops below `theta_merge` **or** structural mass from the cue
  to its clones goes below `edge_eps` (applied in wrapper each episode).

## Quick start

```bash
# install
conda create -n coda-minigrid python=3.10 -y
conda activate coda-minigrid
pip install -r requirements.txt

# train baseline
python train_sb3.py --config configs/ppo_baseline.yaml

# train with CoDA
python train_sb3.py --config configs/ppo_coda.yaml

# evaluate a saved model
python eval_sb3.py --env_id MiniGrid-DoorKey-6x6-v0 --model runs/ppo_coda_MiniGrid-DoorKey-6x6-v0_s0/model.zip --coda
```

Training logs (including CoDA markovization proxies) are written to `runs/.../metrics.csv` with columns:
- `frac_deterministic`
- `norm_entropy`
- `num_states`
- `num_salient`

## Comparing with baselines

- **Baseline (no CoDA)**: PPO receives raw observations only.
- **CoDA**: PPO receives `{"obs": <raw>, "coda": <K-dim feature>}` where the feature includes
  `is_salient` and `prospective` in the first two dimensions and a stable hashed embedding in the rest.

You can train both and compare returns as well as markovization metrics from `metrics.csv`.

## Notes

- This wrapper computes CoDA over **hashed observations** to discrete state ids, which is sufficient
  to drive split/merge online without modifying the underlying environment dynamics.
- If you want to run special protocols (latent inhibition, contingency degradation), plug those
  environments in just like we did in the analysis notebook and keep the wrapper unchanged.
