#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from safetensors import safe_open


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

PATTERN = re.compile(
    r"^(.*\.layers)\.(\d+)\."
    r"(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"\.weight$"
)

METRICS = {
    "relative_change": {
        "title": "Relative weight change from original model",
        "label": "||W(epoch) - W(base)|| / ||W(base)||",
        "cmap": "magma",
    },
    "incremental_change": {
        "title": "Additional weight change during DPO",
        "label": "||W(DPO epoch) - W(SFT final)|| / ||W(SFT final)||",
        "cmap": "magma",
    },
    "direction_cosine": {
        "title": "Alignment of SFT and incremental DPO updates",
        "label": "cos(SFT update, DPO update)",
        "cmap": "coolwarm",
    },
    "epoch_step": {
        "title": "Weight movement during each training epoch",
        "label": "||W(epoch) - W(previous)|| / ||W(previous)||",
        "cmap": "magma",
    },
}


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)

        indices = sorted(self.path.glob("*.safetensors.index.json"))

        if len(indices) > 1:
            raise ValueError(f"Multiple safetensors indices in {self.path}")

        if indices:
            with indices[0].open() as f:
                index = json.load(f)
            self.files = {
                key: self.path / filename
                for key, filename in index["weight_map"].items()
            }
        else:
            candidates = sorted(self.path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(
                    f"No safetensors checkpoint found: {self.path}"
                )
            self.files = {}
            for filename in candidates:
                with safe_open(str(filename), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in self.files:
                            raise ValueError(f"Duplicate tensor: {key}")
                        self.files[key] = filename

        self.handles = {}

    def keys(self):
        return set(self.files)

    def tensor(self, key):
        filename = self.files[key]
        if filename not in self.handles:
            handle = safe_open(
                str(filename), framework="pt", device="cpu"
            )
            handle.__enter__()
            self.handles[filename] = handle
        return self.handles[filename].get_tensor(key).float()

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()


def projection_keys(checkpoint, prefix=None):
    matches = {}
    roots = Counter()

    for key in checkpoint.keys():
        match = PATTERN.match(key)
        if match is None:
            continue

        root, layer, module, projection = match.groups()

        if module == "self_attn" and projection not in PROJECTIONS[:4]:
            continue
        if module == "mlp" and projection not in PROJECTIONS[4:]:
            continue

        roots[root] += 1
        matches[(root, int(layer), projection)] = key

    if prefix is None:
        if not roots:
            raise ValueError(
                f"No supported decoder projection weights in {checkpoint.path}"
            )
        maximum = max(roots.values())
        choices = [name for name, n in roots.items() if n == maximum]
        if len(choices) != 1:
            raise ValueError(
                f"Ambiguous decoder roots: {choices}. Supply --prefix."
            )
        prefix = choices[0]

    selected = {
        (layer, projection): key
        for (root, layer, projection), key in matches.items()
        if root == prefix
    }

    if not selected:
        raise ValueError(f"No projection weights under prefix {prefix}")

    layers = sorted({layer for layer, _ in selected})
    expected = {
        (layer, projection)
        for layer in layers
        for projection in PROJECTIONS
    }

    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise ValueError(
            f"Incomplete decoder projection set for {prefix}: {missing[:12]}"
        )

    if layers != list(range(layers[0], layers[-1] + 1)):
        raise ValueError("Non-contiguous decoder layers")

    return prefix, selected, layers


def norm(x):
    return float(torch.linalg.vector_norm(x))


def relative_delta(a, b):
    return norm(a - b) / max(norm(b), 1e-30)


def update_cosine(a, b):
    an = norm(a)
    bn = norm(b)
    if an == 0 or bn == 0:
        return float("nan")
    return float(
        torch.sum(a.reshape(-1).double() * b.reshape(-1).double())
        / (an * bn)
    )


def analyse_model(name, paths, prefix):
    checkpoints = [Checkpoint(p) for p in paths]

    try:
        base = checkpoints[0]
        root, selected, layers = projection_keys(base, prefix)

        print(f"\n{name}: decoder root = {root}", flush=True)
        print(f"{name}: layers = {layers}", flush=True)

        for index, checkpoint in enumerate(checkpoints[1:], start=1):
            missing = sorted(set(selected.values()) - checkpoint.keys())
            if missing:
                raise ValueError(
                    f"{name} checkpoint {index} missing {len(missing)} "
                    f"decoder tensors. First missing: {missing[0]}"
                )

        rows = []
        count = len(selected)

        for task_index, ((layer, projection), key) in enumerate(
            sorted(selected.items()), start=1
        ):
            weights = [checkpoint.tensor(key) for checkpoint in checkpoints]

            if len({tuple(w.shape) for w in weights}) != 1:
                raise ValueError(f"Shape mismatch for {key}")

            base_w = weights[0]
            final_sft = weights[3]
            sft_update = final_sft - base_w

            for epoch in range(1, 4):
                sft_w = weights[epoch]
                dpo_w = weights[epoch + 3]

                for stage, current, previous in (
                    ("SFT", sft_w, weights[epoch - 1]),
                    ("DPO", dpo_w, weights[epoch + 2]),
                ):
                    rows.append({
                        "model": name,
                        "root": root,
                        "layer": layer,
                        "projection": projection,
                        "stage": stage,
                        "epoch": epoch,
                        "relative_change": relative_delta(current, base_w),
                        "incremental_change": (
                            relative_delta(dpo_w, final_sft)
                            if stage == "DPO" else float("nan")
                        ),
                        "direction_cosine": (
                            update_cosine(sft_update, dpo_w - base_w)
                            if stage == "DPO" else float("nan")
                        ),
                        "epoch_step": relative_delta(current, previous),
                    })

            del weights, sft_update

            if task_index % 14 == 0 or task_index == count:
                print(
                    f"{name}: {task_index}/{count} projection matrices",
                    flush=True,
                )

        return rows

    finally:
        for checkpoint in checkpoints:
            checkpoint.close()


def matrix_for(rows, model, stage, epoch, metric):
    selected = [
        r for r in rows
        if r["model"] == model
        and r["stage"] == stage
        and r["epoch"] == epoch
    ]

    layers = sorted({r["layer"] for r in selected})
    layer_index = {layer: i for i, layer in enumerate(layers)}
    proj_index = {p: i for i, p in enumerate(PROJECTIONS)}

    matrix = np.full((len(PROJECTIONS), len(layers)), np.nan)

    for row in selected:
        matrix[
            proj_index[row["projection"]],
            layer_index[row["layer"]],
        ] = row[metric]

    return matrix, layers


def make_figure(rows, models, metric, output):
    config = METRICS[metric]
    stages = ("SFT", "DPO") if metric != "incremental_change" and metric != "direction_cosine" else ("DPO",)

    panels = [
        (model, stage, epoch)
        for model in models
        for stage in stages
        for epoch in range(1, 4)
    ]

    matrices = {
        panel: matrix_for(rows, *panel, metric)
        for panel in panels
    }

    finite = np.concatenate([
        matrix[np.isfinite(matrix)]
        for matrix, _ in matrices.values()
    ])

    if metric == "direction_cosine":
        norm_obj = Normalize(vmin=-1, vmax=1)
    else:
        positive = finite[finite > 0]
        if positive.size == 0:
            print(f"No nonzero updates for {metric}; skipping")
            return

        norm_obj = LogNorm(
            vmin=max(float(np.percentile(positive, 1)), 1e-10),
            vmax=max(float(np.max(positive)), 1e-9),
        )

    nrows = len(models)
    ncols = len(stages) * 3

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(12, ncols * 3.25), max(4.5, nrows * 3.4)),
        squeeze=False,
        gridspec_kw={"wspace": 0.07, "hspace": 0.30},
    )

    image = None

    for ri, model in enumerate(models):
        for si, stage in enumerate(stages):
            for epoch in range(1, 4):
                ci = si * 3 + epoch - 1
                ax = axes[ri, ci]

                matrix, layers = matrices[(model, stage, epoch)]

                image = ax.imshow(
                    matrix,
                    cmap=config["cmap"],
                    norm=norm_obj,
                    interpolation="nearest",
                    aspect="auto",
                    origin="upper",
                )

                ax.set_title(
                    f"{stage} epoch {epoch}",
                    fontsize=10,
                    fontweight="bold",
                )

                if ci == 0:
                    ax.set_ylabel(
                        f"{model}\n\nProjection",
                        fontsize=10,
                        fontweight="bold",
                    )
                    ax.set_yticks(range(len(PROJECTIONS)))
                    ax.set_yticklabels(PROJECTIONS, fontsize=8)
                else:
                    ax.set_yticks([])

                ticks = np.unique(
                    np.linspace(
                        0, len(layers) - 1,
                        min(5, len(layers)),
                        dtype=int,
                    )
                )
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [str(layers[t]) for t in ticks], fontsize=8
                )
                ax.set_xlabel("Decoder / Thinker layer", fontsize=8)
                ax.tick_params(length=0)

                if si > 0 and epoch == 1:
                    ax.spines["left"].set_color("black")
                    ax.spines["left"].set_linewidth(2.5)

    fig.suptitle(
        config["title"],
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.90,
        top=0.88,
        bottom=0.14,
    )

    colorbar_ax = fig.add_axes([0.925, 0.18, 0.014, 0.63])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(config["label"], fontsize=9)

    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}", flush=True)


def save_csv(rows, path):
    fields = [
        "model", "root", "layer", "projection", "stage", "epoch",
        *METRICS.keys(),
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", nargs=8, action="append", required=True,
        metavar=(
            "NAME", "BASE", "SFT1", "SFT2", "SFT3",
            "DPO1", "DPO2", "DPO3",
        ),
    )
    parser.add_argument(
        "--prefix", action="append", nargs=2,
        metavar=("MODEL", "DECODER_ROOT"),
        help="Explicit decoder root if automatic detection is ambiguous",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/heatmaps"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    prefixes = dict(args.prefix or [])
    names = [spec[0] for spec in args.model]

    if len(names) != len(set(names)):
        parser.error("Model names must be unique")

    if len(names) != 2:
        parser.error("Supply exactly two models for comparative heatmaps")

    all_rows = []

    for spec in args.model:
        name, *paths = spec
        all_rows.extend(
            analyse_model(name, paths, prefixes.get(name))
        )

    save_csv(all_rows, args.output / "weight_update_metrics.csv")

    for metric in METRICS:
        make_figure(
            all_rows,
            names,
            metric,
            args.output / f"{metric}.pdf",
        )

    print("\nCompleted all post-mortem analyses.", flush=True)


if __name__ == "__main__":
    main()
