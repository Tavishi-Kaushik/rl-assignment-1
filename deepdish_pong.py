"""Single-GPU adapter for the existing Assignment1_RL_Submission package.

First run --check. Training requires an explicit --train flag and is NOT run
on import. Keep this file beside verify_submission.py. The original notebook,
loss, baseline, training runner, and saved experiments are not edited.

Execution: CPU Atari simulation and stochastic action sampling; CUDA learner.
Before each episode, the CPU actor receives the learner's current weights.
The learner is unchanged throughout a collected batch, so data remain on-policy.
Only the original Linear/ReLU policy is supported: it has no CUDA-random layers.
Consequently the original checkpoint's CPU RNG state is sufficient for action
sampling on resume. CPU/GPU numerical results need not be bit-identical.

GPU visibility is not a reservation. Follow MLDS allocation rules before use.
--train uses the original runner, including validation-based checkpoint choice
and 500 held-out evaluation episodes AFTER the requested training budget.
It may take hours. Check --help and agree on a budget before starting it.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import weakref

# Must be configured before the first CUDA BLAS operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter(root):
    source = root / "recovery" / "source"
    for name in ("project_code.py", "train_additional.py", "Assignment1_RL_original.ipynb"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing {source / name}; set --root to the extracted package.")
    sys.path.insert(0, str(source))
    import gymnasium as gym
    import ale_py
    from project_code import load_project
    import train_additional as runner

    gym.register_envs(ale_py)
    cpu = load_project()
    cuda = load_project()
    cpu["gym"] = cuda["gym"] = gym
    cuda["pong_device"] = torch.device("cuda:0")
    original_class = cpu["PongPolicy"]
    original_collect = cpu["collect_pong_episode"]
    actors = weakref.WeakKeyDictionary()

    def new_learner():
        model = original_class()
        if any(not isinstance(layer, (torch.nn.Linear, torch.nn.ReLU))
               for layer in model.network):
            raise RuntimeError("This adapter supports only the original deterministic Linear/ReLU policy.")
        return model.to("cuda:0")

    @torch.no_grad()
    def collect_hybrid(env, learner, reset_seed):
        actor = actors.get(learner)
        if actor is None:
            # Its random initialization is immediately replaced. Do not consume
            # the policy-sampling RNG, including when a checkpoint is resumed.
            with torch.random.fork_rng(devices=[]):
                actor = original_class().requires_grad_(False).eval()
            actors[learner] = actor
        # load_state_dict copies values between devices, not parameter objects.
        actor.load_state_dict(learner.state_dict())
        return original_collect(env, actor, reset_seed)

    # Do not change cpu itself: its functions refer to it as their globals.
    adapted = cpu.copy()
    adapted["PongPolicy"] = new_learner
    adapted["collect_pong_episode"] = collect_hybrid
    adapted["accumulate_pong_gradient"] = cuda["accumulate_pong_gradient"]
    return cpu, cuda, adapted, original_class, runner


def flat_gradient(model):
    return torch.cat([p.grad.detach().reshape(-1).cpu() for p in model.parameters()])


def compare_gradient(cpu, cuda, cpu_model, gpu_model, episode, baseline):
    cpu_model.zero_grad(set_to_none=True)
    gpu_model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    cpu_loss = cpu["accumulate_pong_gradient"](
        cpu_model, episode, .99, baseline=baseline, chunk_size=256)
    cpu_seconds = time.perf_counter() - start
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_loss = cuda["accumulate_pong_gradient"](
        gpu_model, episode, .99, baseline=baseline, chunk_size=256)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - start
    expected, measured = flat_gradient(cpu_model), flat_gradient(gpu_model)
    assert torch.isfinite(expected).all() and torch.isfinite(measured).all()
    relative = float(torch.linalg.vector_norm(measured - expected) /
                     torch.linalg.vector_norm(expected).clamp_min(1e-12))
    absolute = float((measured - expected).abs().max())
    assert relative < 5e-4, f"CPU/CUDA gradient relative error: {relative}"
    np.testing.assert_allclose(gpu_loss, cpu_loss, rtol=5e-4, atol=1e-4)
    return dict(cpu_loss=cpu_loss, gpu_loss=gpu_loss,
                relative_gradient_error=relative, maximum_absolute_gradient_error=absolute,
                cpu_gradient_seconds=cpu_seconds, gpu_gradient_seconds=gpu_seconds)


def protocol(root):
    source = root / "recovery" / "source"
    return dict(
        adapter="hybrid_pong_v1", adapter_sha256=digest(__file__),
        project_loader_sha256=digest(source / "project_code.py"),
        training_runner_sha256=digest(source / "train_additional.py"),
        notebook_sha256=digest(source / "Assignment1_RL_original.ipynb"),
        versions={name: importlib.metadata.version(name) for name in
                  ("torch", "numpy", "gymnasium", "ale-py", "matplotlib", "pandas")},
        python=sys.version, cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=True, torch_threads=1,
        rollout_device="cpu", gradient_device="cuda:0",
        objective="-mean_episodes sum_t gamma**t * (G_t-b_t) * log_prob_t",
        gamma=.99, actions=[2, 3], frameskip=4, sticky_action_probability=.25,
        return_normalization=False, point_boundary_return_reset=False,
        original_files_modified=False,
    )


def check(root, cpu, cuda, adapted, original_class):
    torch.manual_seed(80731)
    cpu_model = original_class()
    gpu_model = adapted["PongPolicy"]()
    gpu_model.load_state_dict(cpu_model.state_dict())
    before = {name: value.detach().cpu().clone() for name, value in gpu_model.state_dict().items()}
    generator = np.random.default_rng(80731)
    episode = dict(
        observations=generator.integers(-1, 2, (17, 6400), dtype=np.int8),
        actions=np.arange(17, dtype=np.int64) % 2,
        rewards=np.array([0., 0., -1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., -1.]),
    )
    results = {}
    for name, baseline in (("no_baseline", None),
                           ("time_baseline", np.linspace(-.4, .2, 17))):
        results[name] = compare_gradient(cpu, cuda, cpu_model, gpu_model, episode, baseline)
        print(name, "CPU/CUDA gradient relative error:",
              f"{results[name]['relative_gradient_error']:.3e}", flush=True)

    # A short paired rollout also checks actor weight synchronization, action
    # sampling, frame history, and that the adapter does not change CPU RNG use.
    env_a, env_b = cpu["make_pong_env"](), cpu["make_pong_env"]()
    try:
        torch.manual_seed(90731)
        start = time.perf_counter()
        with torch.no_grad():
            direct = cpu["collect_pong_episode"](env_a, cpu_model, 100_080_731)
        direct_seconds = time.perf_counter() - start
        torch.manual_seed(90731)
        start = time.perf_counter()
        hybrid = adapted["collect_pong_episode"](env_b, gpu_model, 100_080_731)
        hybrid_seconds = time.perf_counter() - start
    finally:
        env_a.close()
        env_b.close()
    for key in ("observations", "actions", "rewards"):
        np.testing.assert_array_equal(direct[key], hybrid[key], err_msg=f"Hybrid rollout differs: {key}")
    assert direct["terminated"] == hybrid["terminated"]
    assert direct["truncated"] == hybrid["truncated"]
    results["real_episode"] = compare_gradient(cpu, cuda, cpu_model, gpu_model, hybrid, None)
    for name, value in gpu_model.state_dict().items():
        assert torch.equal(value.detach().cpu(), before[name]), f"Unexpected parameter change: {name}"
    results["rollout"] = dict(
        steps=len(hybrid["rewards"]), reward=float(hybrid["rewards"].sum()),
        cpu_seconds=direct_seconds, hybrid_seconds=hybrid_seconds,
        exact_actions_rewards_and_frames_match=True, episodes_collected=2,
        environment_seed=100_080_731, action_seed=90731,
    )
    results["protocol"] = protocol(root)
    results["parameter_updates"] = 0
    results["scope"] = "Hardware/adapter correctness and rough timing; not evidence of successful learning."
    parent = root / "deepdish_checks"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="pong_gpu_", dir=parent))
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Paired real-game rollout matched exactly.", flush=True)
    print("Real episode steps:", len(hybrid["rewards"]), flush=True)
    print("Real-episode CPU/CUDA gradient relative error:",
          f"{results['real_episode']['relative_gradient_error']:.3e}", flush=True)
    print("CPU gradient seconds:", round(results["real_episode"]["cpu_gradient_seconds"], 4), flush=True)
    print("GPU gradient seconds:", round(results["real_episode"]["gpu_gradient_seconds"], 4), flush=True)
    print("No training or optimizer updates performed.", flush=True)
    print("Checks saved in:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("deepdish_runs"))
    parser.add_argument("--seeds", type=int, nargs="+", choices=[7, 123], default=[7])
    parser.add_argument("--variants", nargs="+", choices=["no_baseline", "time_baseline"],
                        default=["no_baseline", "time_baseline"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=30)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if min(args.episodes, args.batch_size, args.checkpoint_every, args.validation_episodes) <= 0:
        parser.error("Episode budgets and batch size must be positive.")
    if not np.isfinite(args.lr) or args.lr <= 0 or args.episodes > 100_000:
        parser.error("Use a finite positive learning rate and at most 100000 training episodes.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one permitted GPU with CUDA_VISIBLE_DEVICES before running.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cpu, cuda, adapted, original_class, runner = load_adapter(args.root)
    print("Learner GPU:", torch.cuda.get_device_name(0), "| Rollouts: CPU | Visible GPUs: 1", flush=True)
    if args.check:
        check(args.root, cpu, cuda, adapted, original_class)
        return
    args.game = "pong"
    args.output = (args.root / args.output).resolve()
    protected = [args.root / name for name in ("recovery", "pong_original", "review")]
    if args.output == args.root or any(args.output.is_relative_to(path) for path in protected):
        raise ValueError("Use a separate output folder, not an original-evidence folder.")
    metadata = protocol(args.root)
    check_passed = False
    for path in (args.root / "deepdish_checks").glob("pong_gpu_*/summary.json"):
        evidence = json.loads(path.read_text())
        if (evidence.get("protocol") == metadata and
                evidence.get("parameter_updates") == 0 and
                evidence.get("rollout", {}).get("exact_actions_rewards_and_frames_match") is True):
            check_passed = True
            break
    if not check_passed:
        raise RuntimeError("Run --check successfully with this script and software stack before --train.")
    for seed in args.seeds:
        for variant in args.variants:
            directory = args.output / f"pong_{variant}_seed{seed}"
            metadata_path = directory / "hardware_protocol.json"
            if directory.exists() and any(directory.iterdir()):
                if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != metadata:
                    raise ValueError(f"Different or missing hardware protocol in {directory}; use a new output folder.")
            directory.mkdir(parents=True, exist_ok=True)
            if not metadata_path.exists():
                with metadata_path.open("x") as stream:
                    json.dump(metadata, stream, indent=2)
                    stream.write("\n")
            print("Starting/resuming:", directory, flush=True)
            print("This run includes 500 held-out evaluations after training.", flush=True)
            runner.run_one(args, adapted, seed, variant)


if __name__ == "__main__":
    main()
