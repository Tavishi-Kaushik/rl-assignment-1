"""Audit the completed seed-7 Pong study without training or changing checkpoints.

Default: CSV/config/source-hash and checkpoint ZIP-integrity checks (NumPy only).
--load-checkpoints additionally validates tensors and optimizer state with PyTorch.
--make-plots regenerates presentation plots from the raw observations.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import zipfile

import numpy as np

VARIANTS = ("no_baseline", "time_baseline")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def array(data, key):
    values = np.array([float(row[key]) for row in data], dtype=np.float64)
    assert np.isfinite(values).all(), f"Nonfinite {key}"
    return values


def check_episodes(data, n, environment_start, action_start=None):
    assert len(data) == n, (len(data), n)
    assert [int(r["episode"]) for r in data] == list(range(1, n + 1))
    assert [int(r["environment_seed"]) for r in data] == list(range(environment_start, environment_start + n))
    if action_start is not None:
        assert [int(r["action_seed"]) for r in data] == list(range(action_start, action_start + n))
    reward = array(data, "reward")
    assert np.array_equal(reward, np.round(reward))
    assert np.all((reward >= -21) & (reward <= 21))
    lengths = array(data, "length")
    assert np.all(lengths > 0) and np.array_equal(lengths, lengths.astype(int))
    assert np.all(np.abs(array(data, "discounted_return")) <= 100 + 1e-5)
    for row in data:
        assert row["terminated"] in ("True", "False")
        assert row["truncated"] in ("True", "False")
        assert row["terminated"] == "True" or row["truncated"] == "True"
    return reward


def select_checkpoint(validations):
    # Ascending episode order preserves the earlier checkpoint on an exact tie.
    return max(sorted(validations), key=lambda ep: (
        validations[ep]["mean_reward"], validations[ep]["mean_discounted_return"]))


def describe(scores):
    sd = float(scores.std(ddof=1))
    se = sd / np.sqrt(len(scores))
    mean = float(scores.mean())
    return dict(n=len(scores), mean=mean, sample_sd=sd, se=float(se),
                approximate_95_ci=[mean-1.96*se, mean+1.96*se],
                minimum=float(scores.min()), maximum=float(scores.max()))


def audit(root, load_checkpoints=False):
    root = Path(root).resolve()
    base = root / "deepdish_runs/paired_seed7_b8_e4000"
    source = root / "recovery/source"
    expected_sources = {
        "adapter_sha256": root / "deepdish_pong.py",
        "project_loader_sha256": source / "project_code.py",
        "training_runner_sha256": source / "train_additional.py",
        "notebook_sha256": source / "Assignment1_RL_original.ipynb",
    }
    results = {}
    scores_by_variant = {}
    protocols = []
    configs = []
    for variant in VARIANTS:
        directory = base / f"pong_{variant}_seed7"
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "final_summary.json").read_text())
        protocol = json.loads((directory / "hardware_protocol.json").read_text())
        protocols.append(protocol)
        configs.append({k: v for k, v in config.items() if k not in ("baseline", "loss")})
        expected = dict(game="pong", environment="ALE/Pong-v5", gamma=.99,
                        learning_rate=.001, episodes=4000, batch_size=8,
                        seed=7, baseline=variant, baseline_alpha=.1,
                        checkpoint_every=400, validation_episodes=30,
                        final_episodes=500, return_normalization=False,
                        point_boundary_return_reset=False)
        for k, v in expected.items():
            assert config[k] == v, (variant, k, config[k], v)
        assert all(summary[k] == v for k, v in config.items())
        expected_loss = "-sum_t gamma^t * " + ("(G_t - b_t)" if variant == "time_baseline" else "G_t") + " * log_prob_t"
        assert config["loss"] == expected_loss
        for key, path in expected_sources.items():
            assert sha256(path) == protocol[key], f"Source changed: {path}"
        assert config["source_sha256"] == protocol["notebook_sha256"]
        assert protocol["visible_devices"] == "3" and protocol["gradient_device"] == "cuda:0"
        assert protocol["rollout_device"] == "cpu" and protocol["actions"] == [2, 3]
        assert protocol["frameskip"] == 4 and protocol["sticky_action_probability"] == .25
        assert protocol["gamma"] == .99
        assert protocol["return_normalization"] is False
        assert protocol["point_boundary_return_reset"] is False
        training = rows(directory / "training.csv")
        train_rewards = check_episodes(training, 4000, 20_700_000)
        norms = array(training, "batch_gradient_norm").reshape(-1, 8)
        assert (norms >= 0).all() and np.all(norms == norms[:, :1])
        validation_paths = sorted(directory.glob("validation_*.csv"))
        assert [int(p.stem.split("_")[-1]) for p in validation_paths] == list(range(400, 4001, 400))
        validation = {}
        for path in validation_paths:
            data = rows(path)
            reward = check_episodes(data, 30, 30_007_000, 307_000)
            validation[int(path.stem.split("_")[-1])] = dict(
                mean_reward=float(reward.mean()), sample_sd=float(reward.std(ddof=1)),
                mean_discounted_return=float(array(data, "discounted_return").mean()))
        selected = select_checkpoint(validation)
        assert selected == summary["selected_episode"] == 4000
        final = rows(directory / "final_scores.csv")
        scores = check_episodes(final, 500, 40_007_000, 407_000)
        stats = describe(scores)
        np.testing.assert_allclose([stats["mean"], stats["sample_sd"], stats["se"]],
                                   [summary["mean_reward"], summary["sample_sd"], summary["se"]],
                                   rtol=0, atol=1e-12)
        train_seeds = {int(r["environment_seed"]) for r in training}
        val_seeds = set(range(30_007_000, 30_007_030))
        final_seeds = {int(r["environment_seed"]) for r in final}
        assert train_seeds.isdisjoint(val_seeds | final_seeds)
        assert val_seeds.isdisjoint(final_seeds)
        checkpoints = {}
        for name in ("checkpoint_selected.pt", "checkpoint_latest.pt"):
            path = directory / name
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None, f"Corrupt checkpoint ZIP: {path}"
            checkpoints[name] = dict(sha256=sha256(path), bytes=path.stat().st_size, zip_crc_passed=True)
            if load_checkpoints:
                import torch
                ck = torch.load(path, map_location="cpu", weights_only=True)
                assert ck["config"] == config
                assert ck["completed_episodes"] == 4000
                assert ck["best_score"] == validation[selected]["mean_reward"]
                assert ck["best_return"] == validation[selected]["mean_discounted_return"]
                state = ck["policy_state_dict"]
                shapes = {"network.0.weight": (128, 6400), "network.0.bias": (128,),
                          "network.2.weight": (2, 128), "network.2.bias": (2,)}
                assert set(state) == set(shapes)
                assert sum(v.numel() for v in state.values()) == 819586
                for key, shape in shapes.items():
                    assert tuple(state[key].shape) == shape and torch.isfinite(state[key]).all()
                assert torch.isfinite(ck["time_baseline"]).all()
                assert len(ck["optimizer_state_dict"]["state"]) == 4
                assert all(float(v["step"]) == 500 for v in ck["optimizer_state_dict"]["state"].values())
                for v in ck["optimizer_state_dict"]["state"].values():
                    assert torch.isfinite(v["exp_avg"]).all() and torch.isfinite(v["exp_avg_sq"]).all()
                checkpoints[name]["tensors_and_optimizer_passed"] = True
                if name == "checkpoint_selected.pt":
                    selected_state = state
                else:
                    assert all(torch.equal(selected_state[k], v) for k, v in state.items())
        moving = np.convolve(train_rewards, np.ones(100)/100, "valid")
        stats.update(wins=int((scores > 0).sum()), full_loss_21=int((scores == -21).sum()),
                     terminated=sum(r["terminated"] == "True" for r in final),
                     truncated=sum(r["truncated"] == "True" for r in final),
                     mean_discounted_return=float(array(final, "discounted_return").mean()),
                     mean_episode_length=float(array(final, "length").mean()),
                     reward_counts={str(int(v)): int((scores == v).sum()) for v in np.unique(scores)})
        results[variant] = dict(config=config, selected_episode=selected, final=stats,
            validation=validation, checkpoints=checkpoints, training=dict(
                episodes=len(training), optimizer_updates=500,
                environment_steps=int(array(training, "length").sum()),
                first_100_mean=float(train_rewards[:100].mean()),
                last_100_mean=float(train_rewards[-100:].mean()),
                best_100_mean=float(moving.max()), best_window_end=int(moving.argmax()+100),
                first_100_mean_discounted_return=float(array(training[:100], "discounted_return").mean()),
                last_100_mean_discounted_return=float(array(training[-100:], "discounted_return").mean()),
                truncated=sum(r["truncated"] == "True" for r in training)))
        scores_by_variant[variant] = scores
    assert configs[0] == configs[1] and protocols[0] == protocols[1]
    difference = scores_by_variant["time_baseline"] - scores_by_variant["no_baseline"]
    return dict(passed=True, parameter_updates=0, tensor_loading_performed=load_checkpoints,
                audit_python=platform.python_version(), audit_numpy=np.__version__,
                experiments=results, paired_difference=describe(difference), protocol=protocols[0],
                limitations=["Conditional evaluation intervals, not training-seed uncertainty.",
                    "No fresh gradient-variance experiment for the seed-7 models.",
                    "CSV/source/hash integrity is not proof of successful RL learning.",
                    "Tensor loading is checked only with --load-checkpoints; default checks ZIP CRCs."])


def make_plots(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    base = root / "deepdish_runs/paired_seed7_b8_e4000"
    for variant in VARIANTS:
        directory = base / f"pong_{variant}_seed7"
        title = "No baseline" if variant == "no_baseline" else "Lagged time baseline"
        scores = array(rows(directory/"training.csv"), "reward")
        fig, ax = plt.subplots(figsize=(8.4, 3.6))
        ax.plot(np.arange(1, 4001), scores, color="#7397ae", alpha=.30, lw=.55, label="Episode score")
        ax.plot(np.arange(100, 4001), np.convolve(scores, np.ones(100)/100, "valid"),
                color="#007f83", lw=1.6, label="Trailing 100-episode mean")
        ax.set(title=f"Pong-v5 | {title} | seed 7", xlabel="Training episode", ylabel="Undiscounted episode score",
               xlim=(1, 4000), ylim=(-21.5, -8.5))
        ax.grid(alpha=.18); ax.legend(loc="upper left", frameon=False)
        fig.tight_layout(); fig.savefig(output/f"{variant}_learning_curve.png", dpi=200); plt.close(fig)
        scores = array(rows(directory/"final_scores.csv"), "reward")
        fig, ax = plt.subplots(figsize=(8.4, 3.6))
        ax.hist(scores, bins=np.arange(-21.5, -10.5, 1), color="#007f83", edgecolor="white")
        ax.axvline(scores.mean(), color="#16324a", ls="--", lw=1.5, label=f"Mean = {scores.mean():.3f}")
        ax.set(title=f"Pong-v5 | {title} | 500 held-out episodes", xlabel="Undiscounted episode score",
               ylabel="Number of episodes", xlim=(-21.5, -11.5), ylim=(0, 270), xticks=range(-21, -11))
        ax.grid(axis="y", alpha=.18); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(output/f"{variant}_reward_histogram.png", dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("review/deepdish_audit"))
    parser.add_argument("--load-checkpoints", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    result = audit(root, args.load_checkpoints)
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    if args.make_plots:
        make_plots(root, output)
    for name, data in result["experiments"].items():
        s = data["final"]
        print(f"{name}: training=4000, final N={s['n']}, mean={s['mean']:.3f}, SD={s['sample_sd']:.6f}, selected={data['selected_episode']}")
    print("Paired mean difference and approximate 95% CI:", result["paired_difference"]["mean"],
          result["paired_difference"]["approximate_95_ci"])
    print("Tensor loading performed:", args.load_checkpoints)
    print("Audit passed. No training, evaluation rollouts or optimizer updates performed.")
    print("Evidence:", output)


if __name__ == "__main__":
    main()
