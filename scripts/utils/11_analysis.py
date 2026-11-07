#!/usr/bin/env python3
"""Base-relative SEA/GM weight-space analysis for Omni and Video4B."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from safetensors import safe_open


RUNS = {
    "omni": {
        "sft": {
            "sea": "outputs/_hpo/_work/qwen25_omni/omni_sft_lr4e-6_english80__sea_sft_lr4e-06_e1_seed42",
            "gm": "outputs/_gm/qwen25_omni/omni_gm_sft_lr4e-6_e2",
            "epoch": 2,
        },
        "dpo": {
            "sea": "outputs/_hpo/_work/qwen25_omni/omni_dpo_lr8e-6_english80__sea_dpo_lr8e-06_e1_seed42_beta0.1",
            "gm": "outputs/_gm/qwen25_omni/omni_gm_dpo_lr8e-6_e1",
            "epoch": 1,
        },
    },
    "video4b": {
        "sft": {
            "sea": "outputs/_hpo/_work/video4b/video4b_sea_sft_lr2e-6_english80__sea_sft_lr2e-06_e1_seed42",
            "gm": "outputs/_gm/video4b/video4b_gm_sft_lr2e-6_e1",
            "epoch": 1,
        },
        "dpo": {
            "sea": "outputs/_hpo/_work/video4b/video4b_sea_dpo_lr4e-6_english80__sea_dpo_lr4e-06_e1_seed42_beta0.1",
            "gm": "outputs/_gm/video4b/video4b_gm_dpo_lr4e-6_e1",
            "epoch": 1,
        },
    },
}

PROJECTORS = {
    "omni": "audio_tower.proj.weight",
    "video4b": "model.audio.audio_proj.weight",
}

DECODER_PREFIXES = {
    "omni": "model.layers.",
    "video4b": "model.language_model.layers.",
}

DECODER_MATRICES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}

NORM_FIELDS = [
    "model", "method", "variant", "group", "layer", "matrix",
    "parameter", "base_norm", "update_norm", "relative_change",
]

DIRECTION_FIELDS = [
    "model", "method", "group", "layer", "matrix", "parameter",
    "update_cosine", "input_top1_abs_cosine",
    "output_top1_abs_cosine", "input_top4_subspace",
    "output_top4_subspace",
]


def selected_matrix(model, key):
    if key == PROJECTORS[model]:
        return "projector", -1, "audio_proj"

    prefix = DECODER_PREFIXES[model]
    if not key.startswith(prefix) or not key.endswith(".weight"):
        return None

    match = re.fullmatch(
        re.escape(prefix)
        + r"(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight",
        key,
    )
    if match is None or match.group(2) not in DECODER_MATRICES:
        return None

    return "decoder", int(match.group(1)), match.group(2)


def read_config(directory):
    path = directory / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def checkpoint(directory, epoch):
    path = directory / "checkpoints" / f"epoch_{epoch:03d}" / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def index_base(directory):
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    mapping = json.loads(index_path.read_text())["weight_map"]
    result = {key: directory / filename for key, filename in mapping.items()}

    for path in set(result.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    return result


def base_key(model, key):
    if model == "omni":
        return "thinker." + key
    return key


def read_tensor(path, key):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(torch.float32)


def prepare(root):
    prepared = {}
    signatures = []

    for model, methods in RUNS.items():
        for method, config in methods.items():
            paths = {
                variant: root / config[variant]
                for variant in ("sea", "gm")
            }

            configs = {
                variant: read_config(path)
                for variant, path in paths.items()
            }

            base_paths = [
                Path(configs[variant]["model_path"]).resolve()
                for variant in ("sea", "gm")
            ]

            if base_paths[0] != base_paths[1]:
                raise ValueError(f"Different training bases: {model}/{method}")

            base_dir = base_paths[0]
            base_map = index_base(base_dir)

            checkpoints = {
                variant: checkpoint(paths[variant], config["epoch"])
                for variant in ("sea", "gm")
            }

            key_sets = {}
            for variant, path in checkpoints.items():
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    key_sets[variant] = {
                        key for key in handle.keys()
                        if selected_matrix(model, key) is not None
                    }

            if key_sets["sea"] != key_sets["gm"]:
                raise ValueError(f"SEA/GM matrix mismatch: {model}/{method}")

            keys = sorted(key_sets["sea"])
            if not keys or PROJECTORS[model] not in keys:
                raise ValueError(f"Missing projector: {model}/{method}")

            decoder_keys = [
                key for key in keys
                if selected_matrix(model, key)[0] == "decoder"
            ]
            if not decoder_keys:
                raise ValueError(f"Missing decoder matrices: {model}/{method}")

            for key in keys:
                mapped = base_key(model, key)
                if mapped not in base_map:
                    raise KeyError(f"Base tensor missing: {mapped}")

            for path in [*checkpoints.values(), *set(base_map.values())]:
                stat = path.stat()
                signatures.append((
                    str(path.resolve()), stat.st_size, stat.st_mtime_ns
                ))

            prepared[(model, method)] = {
                "keys": keys,
                "base": base_map,
                "checkpoints": checkpoints,
            }

            print(
                f"{model}/{method}: {len(keys)} matrices; "
                f"base={base_dir}",
                flush=True,
            )

    return prepared, signatures


def frobenius(tensor):
    return float(torch.linalg.vector_norm(tensor))


def update_cosine(a, b):
    denominator = frobenius(a) * frobenius(b)
    if denominator == 0:
        return float("nan")
    return float(torch.sum(a * b) / denominator)


def svd_basis(matrix, rank, device, seed):
    if not torch.count_nonzero(matrix):
        return None

    rank = min(rank, *matrix.shape)
    q = min(min(matrix.shape), max(rank + 4, 2 * rank))

    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()] if device.type == "cuda" else []
    ):
        torch.manual_seed(seed)

        with torch.no_grad():
            u, _, v = torch.svd_lowrank(
                matrix.to(device),
                q=q,
                niter=2,
            )

    return u[:, :rank].cpu(), v[:, :rank].cpu()


def basis_similarity(a, b):
    if a is None or b is None:
        return float("nan"), float("nan")

    top1 = abs(float(torch.dot(a[:, 0], b[:, 0])))

    cosines = torch.linalg.svdvals(a.T @ b).clamp(0, 1)
    subspace = float(torch.sqrt(torch.mean(cosines.square())))

    return top1, subspace


def analyse_matrix(model, method, key, item, device, rank, seed):
    group, layer, matrix_name = selected_matrix(model, key)
    mapped = base_key(model, key)

    base = read_tensor(item["base"][mapped], mapped)
    sea = read_tensor(item["checkpoints"]["sea"], key)
    gm = read_tensor(item["checkpoints"]["gm"], key)

    if base.shape != sea.shape or base.shape != gm.shape:
        raise ValueError(f"Tensor shape mismatch: {key}")
    if base.ndim != 2:
        raise ValueError(f"Expected matrix, got {tuple(base.shape)}: {key}")

    delta_sea = sea - base
    delta_gm = gm - base

    base_norm = frobenius(base)
    if base_norm == 0:
        raise ValueError(f"Zero base norm: {key}")

    norm_rows = []
    for variant, delta in (("sea", delta_sea), ("gm", delta_gm)):
        magnitude = frobenius(delta)
        norm_rows.append({
            "model": model,
            "method": method,
            "variant": variant,
            "group": group,
            "layer": layer,
            "matrix": matrix_name,
            "parameter": key,
            "base_norm": base_norm,
            "update_norm": magnitude,
            "relative_change": magnitude / base_norm,
        })

    full_cosine = update_cosine(delta_sea, delta_gm)

    left_sea, right_sea = (None, None)
    left_gm, right_gm = (None, None)

    sea_basis = svd_basis(delta_sea, rank, device, seed)
    gm_basis = svd_basis(delta_gm, rank, device, seed)

    if sea_basis is not None:
        left_sea, right_sea = sea_basis
    if gm_basis is not None:
        left_gm, right_gm = gm_basis

    input_top1, input_subspace = basis_similarity(right_sea, right_gm)
    output_top1, output_subspace = basis_similarity(left_sea, left_gm)

    direction_row = {
        "model": model,
        "method": method,
        "group": group,
        "layer": layer,
        "matrix": matrix_name,
        "parameter": key,
        "update_cosine": full_cosine,
        "input_top1_abs_cosine": input_top1,
        "output_top1_abs_cosine": output_top1,
        "input_top4_subspace": input_subspace,
        "output_top4_subspace": output_subspace,
    }

    return norm_rows, direction_row


def load_progress(path, signature):
    if not path.is_file():
        return {}

    completed = {}
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break

            if record["signature"] != signature:
                raise ValueError("Checkpoint inputs changed; progress is stale")

            identity = (
                record["model"], record["method"], record["parameter"]
            )
            completed[identity] = record

    return completed


def write_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_grid(rows, field, layers, matrices):
    lookup = {
        (int(row["layer"]), row["matrix"]): float(row[field])
        for row in rows
    }

    return np.array([
        [lookup.get((layer, matrix), np.nan) for matrix in matrices]
        for layer in layers
    ])



def create_pdf(path, norm_rows, direction_rows):
    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_order = [
        "audio_proj", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    for model in ("omni", "video4b"):
        for method in ("sft", "dpo"):
            nr = [
                row for row in norm_rows
                if row["model"] == model and row["method"] == method
            ]
            dr = [
                row for row in direction_rows
                if row["model"] == model and row["method"] == method
            ]

            layers = [-1] + sorted({
                int(row["layer"]) for row in dr
                if int(row["layer"]) >= 0
            })
            matrices = [
                matrix for matrix in matrix_order
                if any(row["matrix"] == matrix for row in dr)
            ]

            sea_rows = [row for row in nr if row["variant"] == "sea"]
            gm_rows = [row for row in nr if row["variant"] == "gm"]

            shared_max = max(
                max(float(row["relative_change"]) for row in nr),
                1e-12,
            )

            panels = [
                ("SEA relative change", sea_rows,
                 "relative_change", 0, shared_max, "viridis"),
                ("GM relative change", gm_rows,
                 "relative_change", 0, shared_max, "viridis"),
                ("SEA vs GM update cosine", dr,
                 "update_cosine", -1, 1, "coolwarm"),
            ]

            fig, axes = plt.subplots(
                3, 1, figsize=(19, 10),
                constrained_layout=True,
            )
            fig.suptitle(
                f"{model.upper()} — {method.upper()}",
                fontsize=17,
            )

            for ax, (title, rows, field, vmin, vmax, cmap) in zip(
                axes, panels
            ):
                # Transpose so matrices are rows and layers are columns.
                grid = make_grid(rows, field, layers, matrices).T

                image = ax.imshow(
                    np.ma.masked_invalid(grid),
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                ax.set_title(title)
                ax.set_xticks(range(len(layers)))
                ax.set_xticklabels([
                    "Proj" if layer == -1 else str(layer)
                    for layer in layers
                ])
                ax.set_yticks(range(len(matrices)))
                ax.set_yticklabels(matrices)
                ax.set_xlabel("Projector / Decoder layer")
                ax.set_ylabel("Weight matrix")

                fig.colorbar(
                    image, ax=ax, fraction=0.015, pad=0.01
                )

            output_path = output_dir / f"{model}_{method}.pdf"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {output_path}", flush=True)

    if path.exists():
        path.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.rank < 1:
        parser.error("--rank must be positive")

    torch.set_grad_enabled(False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}", flush=True)

    root = args.root.resolve()
    prepared, input_signatures = prepare(root)

    signature = hashlib.sha256(
        json.dumps(
            [input_signatures, args.rank, args.seed],
            sort_keys=True,
        ).encode()
    ).hexdigest()

    output = root / "results/optimised_params/weight-space"
    data_dir = output / "data"
    heatmap_dir = output / "heatmaps"
    progress_dir = root / "outputs/11_analysis"

    for directory in (data_dir, heatmap_dir, progress_dir):
        directory.mkdir(parents=True, exist_ok=True)

    journal = progress_dir / "weight_space_progress.jsonl"
    completed = load_progress(journal, signature)

    tasks = [
        (model, method, key, item)
        for (model, method), item in prepared.items()
        for key in item["keys"]
    ]

    print(
        f"Matrices: {len(tasks)}, previously completed: {len(completed)}",
        flush=True,
    )

    with journal.open("a") as file:
        for index, (model, method, key, item) in enumerate(tasks, 1):
            identity = (model, method, key)
            if identity in completed:
                continue

            norms, direction = analyse_matrix(
                model, method, key, item,
                device, args.rank, args.seed,
            )

            record = {
                "signature": signature,
                "model": model,
                "method": method,
                "parameter": key,
                "norms": norms,
                "direction": direction,
            }

            file.write(json.dumps(record, allow_nan=True) + "\n")
            file.flush()
            completed[identity] = record

            if index % 10 == 0 or index == len(tasks):
                print(
                    f"{index}/{len(tasks)}: {model}/{method}, "
                    f"{key}",
                    flush=True,
                )

    norm_rows = []
    direction_rows = []

    for model, method, key, _ in tasks:
        record = completed[(model, method, key)]
        norm_rows.extend(record["norms"])
        direction_rows.append(record["direction"])

    write_csv(
        data_dir / "relative_changes.csv",
        norm_rows,
        NORM_FIELDS,
    )
    write_csv(
        data_dir / "update_directions.csv",
        direction_rows,
        DIRECTION_FIELDS,
    )
    create_pdf(
        heatmap_dir / "weight_space_analysis.pdf",
        norm_rows,
        direction_rows,
    )

    print(f"Completed: {len(tasks)} matrices", flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
