
import argparse, pandas as pd, matplotlib.pyplot as plt
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", type=str)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.metrics_csv)
    fig, axes = plt.subplots(3,1, figsize=(7,7), sharex=True)
    axes[0].plot(df["timesteps"], df["frac_deterministic"], label="frac_deterministic")
    axes[0].plot(df["timesteps"], 1-df["norm_entropy"], label="1 - norm_entropy")
    axes[0].set_ylabel("Markovianity"); axes[0].legend()
    axes[1].plot(df["timesteps"], df["coda_space_bytes"]); axes[1].set_ylabel("CoDA space (B)")
    if "success_probe" in df.columns:
        axes[2].plot(df["timesteps"], df["success_probe"]); axes[2].set_ylabel("success (probe)")
    axes[2].set_xlabel("timesteps")
    fig.tight_layout()
    if args.out: fig.savefig(args.out, dpi=150)
    else: plt.show()
if __name__ == "__main__":
    main()
