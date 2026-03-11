from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_ieee_style() -> None:
    # IEEE-friendly defaults (single-column readable figure settings).
    plt.rcParams.update(
        {
            "figure.figsize": (3.5, 2.4),
            "figure.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "lines.linewidth": 1.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_losses(loss_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with loss_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    train = np.array([d["train_loss"] for d in data], dtype=float)
    val = np.array([d["val_loss"] for d in data], dtype=float)
    return train, val


def min_val_loss(loss_path: Path) -> float:
    _, val = read_losses(loss_path)
    return float(np.nanmin(val))


def scheduled_sparsity_epoch(
    epoch: int,
    total_epochs: int,
    prune_start_epoch: int,
    prune_end_epoch: int,
    target_sparsity: float,
    schedule: str,
) -> float:
    # Epoch-wise version of the schedule used in train_imp.py.
    if epoch < prune_start_epoch:
        return 0.0
    if epoch > prune_end_epoch:
        return target_sparsity

    total_prune_epochs = prune_end_epoch - prune_start_epoch + 1
    progress = (epoch - prune_start_epoch + 1) / total_prune_epochs
    progress = float(np.clip(progress, 0.0, 1.0))

    if schedule == "linear":
        return target_sparsity * progress
    if schedule == "exponential":
        return 1.0 - ((1.0 - target_sparsity) ** progress)
    raise ValueError(f"Unsupported schedule: {schedule}")


def plot_prune_scheduler() -> None:
    total_epochs = 1500
    start_epoch = 10
    end_epoch = 750
    target = 0.90
    epochs = np.arange(1, total_epochs + 1)

    linear = np.array(
        [
            scheduled_sparsity_epoch(
                epoch=e,
                total_epochs=total_epochs,
                prune_start_epoch=start_epoch,
                prune_end_epoch=end_epoch,
                target_sparsity=target,
                schedule="linear",
            )
            for e in epochs
        ]
    )
    exp = np.array(
        [
            scheduled_sparsity_epoch(
                epoch=e,
                total_epochs=total_epochs,
                prune_start_epoch=start_epoch,
                prune_end_epoch=end_epoch,
                target_sparsity=target,
                schedule="exponential",
            )
            for e in epochs
        ]
    )

    fig, ax = plt.subplots()
    ax.plot(
        epochs,
        exp * 100.0,
        label=r"Exponential ($e_{\mathrm{start}}=10$, $e_{\mathrm{end}}=750$, $s_{\max}=90\%$)",
        color="tab:blue",
    )
    ax.plot(
        epochs,
        linear * 100.0,
        label=r"Linear ($e_{\mathrm{start}}=10$, $e_{\mathrm{end}}=750$, $s_{\max}=90\%$)",
        color="tab:orange",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparsity (%)")
    ax.set_xlim(1, total_epochs)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(OUT_DIR / "prune_scheduler_ieee.png")
    plt.close(fig)


def plot_loss_curves() -> None:
    models = {
        "Global Exponential": ROOT
        / "models/prune_type_schedule/output-b40-lr0.001-e1500-p90-global-exponential-ps10-pe750-2026-03-09_12-10-42-834574/losses.json",
        "Global Linear": ROOT
        / "models/prune_type_schedule/output-b40-lr0.001-e1500-p90-global-linear-ps10-pe750-2026-03-09_13-35-08-058726/losses.json",
        "Local Exponential": ROOT
        / "models/prune_type_schedule/output-b40-lr0.001-e1500-p90-local-exponential-ps10-pe750-2026-03-09_15-03-46-695406/losses.json",
        "Local Linear": ROOT
        / "models/prune_type_schedule/output-b40-lr0.001-e1500-p90-local-linear-ps10-pe750-2026-03-09_16-23-14-703120/losses.json",
    }
    colors = {
        "Global Exponential": "tab:blue",
        "Global Linear": "tab:orange",
        "Local Exponential": "tab:green",
        "Local Linear": "tab:red",
    }

    fig, ax = plt.subplots(figsize=(9.0, 3))
    for model_name, path in models.items():
        train, val = read_losses(path)
        epochs = np.arange(1, len(train) + 1)
        ax.plot(
            epochs,
            train,
            color=colors[model_name],
            linestyle="-",
            linewidth=0.7,
            label=model_name,
        )
        ax.plot(
            epochs,
            val,
            color=colors[model_name],
            linestyle=":",
            linewidth=0.7,
            label="_nolegend_",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ESR")
    ax.set_xlim(1, 1500)
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=2, frameon=True)
    fig.savefig(OUT_DIR / "loss_curves_ieee.png")
    plt.close(fig)


def collect_iterative_sparsity_curve() -> tuple[np.ndarray, np.ndarray]:
    base = ROOT / "models" / "sparsity_level_sweep"
    sparse_to_loss: dict[int, float] = {}
    for run_dir in sorted(base.glob("output-b40-lr0.001-e1500*")):
        if not run_dir.is_dir():
            continue

        losses_path = run_dir / "losses.json"
        if not losses_path.exists():
            continue

        match = re.search(r"-p(\d+)-", run_dir.name)
        sparsity = int(match.group(1)) if match else 0
        sparse_to_loss[sparsity] = min_val_loss(losses_path)

    sparsities = np.array(sorted(sparse_to_loss.keys()), dtype=float)
    losses = np.array([sparse_to_loss[int(s)] for s in sparsities], dtype=float)
    return sparsities, losses


def collect_one_shot_sparsity_curve(zero_percent_loss: float) -> tuple[np.ndarray, np.ndarray]:
    summary_path = ROOT / "models" / "one_shot_sweep" / "prune_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    sparse_to_loss: dict[int, float] = {0: zero_percent_loss}
    for item in summary:
        target = float(item["target_sparsity"]) * 100.0
        sparse_to_loss[int(round(target))] = float(item["val_loss"])

    sparsities = np.array(sorted(sparse_to_loss.keys()), dtype=float)
    losses = np.array([sparse_to_loss[int(s)] for s in sparsities], dtype=float)
    return sparsities, losses


def plot_sparsity_sweep() -> None:
    it_s, it_l = collect_iterative_sparsity_curve()
    zero_loss = float(it_l[np.where(it_s == 0)[0][0]])
    os_s, os_l = collect_one_shot_sparsity_curve(zero_percent_loss=zero_loss)

    fig, ax = plt.subplots()
    ax.plot(it_s, it_l, marker="o", color="tab:blue", label="Iterative")
    ax.plot(os_s, os_l, marker="s", color="tab:orange", label="One-shot")
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("ESR")
    ax.set_xlim(0, 100)
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(OUT_DIR / "sparsity_sweep_ieee.png")
    plt.close(fig)


def _read_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=float), int(sample_rate)


def plot_waveform_overlap(
    model_output_wav: Path,
    target_wav: Path,
    output_png: Path,
    start_sec: float = 0.7,
    window_sec: float = 0.008,
) -> None:
    model_audio, sr_model = _read_mono_audio(model_output_wav)
    target_audio, sr_target = _read_mono_audio(target_wav)
    if sr_model != sr_target:
        raise ValueError(f"Sample-rate mismatch: {sr_model} vs {sr_target}")

    start_idx = int(round(start_sec * sr_model))
    end_idx = start_idx + int(round(window_sec * sr_model))
    n = min(len(model_audio), len(target_audio))
    if end_idx > n:
        raise ValueError(f"Requested window exceeds available audio length: {n / sr_model:.3f}s")

    model_seg = model_audio[start_idx:end_idx]
    target_seg = target_audio[start_idx:end_idx]
    time_ms = (np.arange(len(model_seg)) / sr_model) * 1000.0

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.plot(time_ms, target_seg, color="tab:blue", linewidth=1.0, label="Target")
    ax.plot(time_ms, model_seg, color="tab:orange", linewidth=1.0, linestyle="--", label="Model output")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(output_png)
    plt.close(fig)


def plot_waveform_overlaps() -> None:
    plot_waveform_overlap(
        model_output_wav=ROOT
        / "models/sparsity_level_sweep/output-b40-lr0.001-e1500-p90-local-exponential-ps10-pe750-2026-03-10_10-41-36-427138/model_output.wav",
        target_wav=ROOT
        / "models/sparsity_level_sweep/output-b40-lr0.001-e1500-p90-local-exponential-ps10-pe750-2026-03-10_10-41-36-427138/target.wav",
        output_png=OUT_DIR / "waveform_overlap_imp_p90_ieee.png",
    )
    plot_waveform_overlap(
        model_output_wav=ROOT
        / "models/one_shot_sweep/p90/model_output.wav",
        target_wav=ROOT
        / "models/one_shot_sweep/p90/target.wav",
        output_png=OUT_DIR / "waveform_overlap_oneshot_p90_high_ieee.png",
    )
    plot_waveform_overlap(
        model_output_wav=ROOT
        / "models/sparsity_level_sweep/output-b40-lr0.001-e1500-2026-03-09_18-18-33-026461/model_output.wav",
        target_wav=ROOT
        / "models/sparsity_level_sweep/output-b40-lr0.001-e1500-2026-03-09_18-18-33-026461/target.wav",
        output_png=OUT_DIR / "waveform_overlap_not_pruned_high_ieee.png",
    )


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def build_amp_capture_rows() -> list[tuple[str, float]]:
    base = ROOT / "models" / "amp_captures"
    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    rows: list[tuple[str, float]] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        losses_path = run_dir / "losses.json"
        if not losses_path.exists():
            continue

        # Model name is the first token before the first '-'.
        model_name = run_dir.name.split("-", maxsplit=1)[0]
        rows.append((model_name, min_val_loss(losses_path)))

    rows.sort(key=lambda x: x[0].lower())
    return rows


def make_latex_esr_table(rows: list[tuple[str, float]]) -> str:
    if not rows:
        raise ValueError("No rows available to create LaTeX table.")

    split_idx = (len(rows) + 1) // 2
    left = rows[:split_idx]
    right = rows[split_idx:]
    while len(right) < len(left):
        right.append(("", float("nan")))

    lines = [
        r"\begin{table}[!t]",
        r"\caption{Minimum validation ESR for amplifier capture models (split into two column blocks).}",
        r"\label{tab:amp_capture_min_esr}",
        r"\centering",
        r"\begin{tabular}{|l|c||l|c|}",
        r"\hline",
        r"Model & ESR & Model & ESR \\",
        r"\hline",
    ]

    for (m1, e1), (m2, e2) in zip(left, right):
        left_model = _latex_escape(m1) if m1 else "--"
        left_esr = f"{e1:.2e}" if np.isfinite(e1) else "--"
        right_model = _latex_escape(m2) if m2 else "--"
        right_esr = f"{e2:.2e}" if np.isfinite(e2) else "--"
        lines.append(f"{left_model} & {left_esr} & {right_model} & {right_esr} \\\\")
        lines.append(r"\hline")

    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    setup_ieee_style()
    plot_prune_scheduler()
    plot_loss_curves()
    plot_sparsity_sweep()
    plot_waveform_overlaps()
    amp_rows = build_amp_capture_rows()
    latex_table = make_latex_esr_table(amp_rows)
    print(f"Saved plots to: {OUT_DIR}")
    print()
    print("LaTeX table (copy-paste):")
    print(latex_table)


if __name__ == "__main__":
    main()
