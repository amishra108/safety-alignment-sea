#!/usr/bin/env python3

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
)


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
MODEL_PATH = ROOT / "models/qwen2.5-omni-7b"
EMBED_ROOT = ROOT / "outputs/02_optimised_embeddings/qwen25_omni"
MANIFEST = ROOT / "data/eval/audio_jailbreaks_english_200_seed42.csv"
OUT = ROOT / "outputs/diagnostics/synthetic_vs_real_qwen/projector_input_output"

DEVICE = torch.device("cuda:0")
REAL_INSTRUCTION = "Please respond to the request in the audio."
SEED = 42
N_SYNTHETIC = 200


def summary(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def pool(x):
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 2:
        raise RuntimeError(f"Expected 2D sequence, got {tuple(x.shape)}")

    return x.mean(dim=0)


def token_norms(sequences):
    values = []

    for sequence in sequences:
        x = sequence

        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]

        if x.ndim != 2:
            raise RuntimeError(
                f"Expected 2D sequence for token norms, got {tuple(x.shape)}"
            )

        values.extend(
            torch.linalg.vector_norm(
                x.float(),
                dim=-1,
            ).cpu().numpy().tolist()
        )

    return np.asarray(values, dtype=np.float64)


def normalise_rows(x):
    return x / (
        np.linalg.norm(
            x,
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )


def domain_classifier(synthetic, real, name):
    X = np.concatenate(
        [synthetic, real],
        axis=0,
    )

    y = np.concatenate(
        [
            np.zeros(len(synthetic), dtype=np.int64),
            np.ones(len(real), dtype=np.int64),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    aurocs = []
    balanced = []

    print()
    print(f"===== DOMAIN CLASSIFIER: {name} =====")

    for fold, (train, test) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X[train]
        )
        X_test = scaler.transform(
            X[test]
        )

        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        )

        clf.fit(
            X_train,
            y[train],
        )

        prob = clf.predict_proba(
            X_test
        )[:, 1]

        pred = clf.predict(
            X_test
        )

        auc = roc_auc_score(
            y[test],
            prob,
        )

        bacc = balanced_accuracy_score(
            y[test],
            pred,
        )

        aurocs.append(float(auc))
        balanced.append(float(bacc))

        print(
            f"fold={fold} "
            f"AUROC={auc:.4f} "
            f"balanced_accuracy={bacc:.4f}",
            flush=True,
        )

    result = {
        "aurocs": aurocs,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "balanced_accuracies": balanced,
        "balanced_accuracy_mean": float(np.mean(balanced)),
        "balanced_accuracy_std": float(np.std(balanced)),
    }

    print(
        f"mean AUROC={result['auroc_mean']:.4f} "
        f"std={result['auroc_std']:.4f}"
    )
    print(
        f"mean balanced_accuracy="
        f"{result['balanced_accuracy_mean']:.4f} "
        f"std={result['balanced_accuracy_std']:.4f}"
    )

    return result


def compare_space(
    name,
    synthetic_sequences,
    real_sequences,
):
    synthetic_pooled = np.stack(
        [
            pool(x).cpu().numpy()
            for x in synthetic_sequences
        ],
        axis=0,
    )

    real_pooled = np.stack(
        [
            pool(x).cpu().numpy()
            for x in real_sequences
        ],
        axis=0,
    )

    synth_token_norm = token_norms(
        synthetic_sequences
    )
    real_token_norm = token_norms(
        real_sequences
    )

    synth_pool_norm = np.linalg.norm(
        synthetic_pooled,
        axis=1,
    )
    real_pool_norm = np.linalg.norm(
        real_pooled,
        axis=1,
    )

    synth_mean = synthetic_pooled.mean(
        axis=0
    )
    real_mean = real_pooled.mean(
        axis=0
    )

    mean_l2 = float(
        np.linalg.norm(
            synthetic_mean - real_mean
        )
    )

    mean_cos = float(
        np.dot(
            synthetic_mean,
            real_mean,
        )
        / (
            np.linalg.norm(synthetic_mean)
            * np.linalg.norm(real_mean)
            + 1e-12
        )
    )

    synth_unit = normalise_rows(
        synthetic_pooled
    )
    real_unit = normalise_rows(
        real_pooled
    )

    cross_cos = (
        real_unit
        @ synth_unit.T
    )

    real_best = cross_cos.max(
        axis=1
    )
    synth_best = cross_cos.max(
        axis=0
    )

    synthetic_dim_mean = (
        synthetic_pooled.mean(axis=0)
    )
    real_dim_mean = (
        real_pooled.mean(axis=0)
    )

    synthetic_dim_std = (
        synthetic_pooled.std(axis=0)
    )
    real_dim_std = (
        real_pooled.std(axis=0)
    )

    dim_mean_rmse = float(
        np.sqrt(
            np.mean(
                (
                    synthetic_dim_mean
                    - real_dim_mean
                ) ** 2
            )
        )
    )

    dim_std_rmse = float(
        np.sqrt(
            np.mean(
                (
                    synthetic_dim_std
                    - real_dim_std
                ) ** 2
            )
        )
    )

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print()
    print("PER-TOKEN NORMS")
    print(
        "synthetic:",
        summary(synth_token_norm),
    )
    print(
        "real     :",
        summary(real_token_norm),
    )

    print()
    print("POOLED VECTOR NORMS")
    print(
        "synthetic:",
        summary(synth_pool_norm),
    )
    print(
        "real     :",
        summary(real_pool_norm),
    )

    print()
    print("POPULATION MEANS")
    print(
        "L2(real_mean - synthetic_mean):",
        mean_l2,
    )
    print(
        "cos(real_mean, synthetic_mean):",
        mean_cos,
    )

    print()
    print("DIMENSION-WISE DISTRIBUTION DIFFERENCE")
    print(
        "mean RMSE:",
        dim_mean_rmse,
    )
    print(
        "std RMSE :",
        dim_std_rmse,
    )

    print()
    print("CROSS-DOMAIN NEAREST COSINE")
    print(
        "real -> best synthetic:",
        summary(real_best),
    )
    print(
        "synthetic -> best real:",
        summary(synth_best),
    )

    classifier = domain_classifier(
        synthetic_pooled,
        real_pooled,
        name,
    )

    return {
        "synthetic_token_norms":
            summary(synth_token_norm),
        "real_token_norms":
            summary(real_token_norm),
        "synthetic_pooled_norms":
            summary(synth_pool_norm),
        "real_pooled_norms":
            summary(real_pool_norm),
        "population_mean_l2":
            mean_l2,
        "population_mean_cosine":
            mean_cos,
        "dimension_mean_rmse":
            dim_mean_rmse,
        "dimension_std_rmse":
            dim_std_rmse,
        "real_to_synthetic_best_cosine":
            summary(real_best),
        "synthetic_to_real_best_cosine":
            summary(synth_best),
        "classifier":
            classifier,
    }


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "Loading Qwen2.5-Omni...",
    flush=True,
)

processor = (
    Qwen2_5OmniProcessor
    .from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
)

dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

model = (
    Qwen2_5OmniThinkerForConditionalGeneration
    .from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        local_files_only=True,
    )
    .to(DEVICE)
)

model.eval()

encoders = [
    module
    for module in model.modules()
    if isinstance(
        module,
        Qwen2_5OmniAudioEncoder,
    )
]

if len(encoders) != 1:
    raise RuntimeError(
        "Expected exactly one audio encoder, "
        f"found {len(encoders)}"
    )

projector = encoders[0].proj

print(
    "Boundary: Qwen2_5OmniAudioEncoder.proj",
    flush=True,
)


# ------------------------------------------------------------------
# Synthetic
# ------------------------------------------------------------------

rng = random.Random(SEED)

synthetic_indices = sorted(
    rng.sample(
        range(2000),
        N_SYNTHETIC,
    )
)

synthetic_inputs = []
synthetic_outputs = []

with torch.no_grad():
    for position, index in enumerate(
        synthetic_indices,
        start=1,
    ):
        path = (
            EMBED_ROOT
            / f"row_{index:04d}.pt"
        )

        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        x = record["embedding"].float()

        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]

        if x.ndim != 2:
            raise RuntimeError(
                f"Synthetic row {index}: "
                f"unexpected shape {tuple(x.shape)}"
            )

        y = projector(
            x.to(
                device=DEVICE,
                dtype=dtype,
            )
        )

        if isinstance(y, (tuple, list)):
            if len(y) != 1:
                raise RuntimeError(
                    f"Synthetic row {index}: "
                    "unexpected projector tuple output"
                )
            y = y[0]

        if not torch.is_tensor(y):
            raise RuntimeError(
                f"Synthetic row {index}: "
                "projector did not return tensor"
            )

        synthetic_inputs.append(
            x.cpu()
        )
        synthetic_outputs.append(
            y.detach().float().cpu()
        )

        if (
            position == 1
            or position % 25 == 0
            or position == N_SYNTHETIC
        ):
            print(
                f"Synthetic {position}/{N_SYNTHETIC} "
                f"row={index} "
                f"in={tuple(x.shape)} "
                f"out={tuple(y.shape)}",
                flush=True,
            )


# ------------------------------------------------------------------
# Real audio
# ------------------------------------------------------------------

with MANIFEST.open(
    encoding="utf-8",
    newline="",
) as f:
    rows = list(
        csv.DictReader(f)
    )

if len(rows) != 200:
    raise RuntimeError(
        f"Expected 200 validation rows, got {len(rows)}"
    )

real_inputs = []
real_outputs = []

capture_input = []
capture_output = []


def pre_hook(_module, inputs):
    if not inputs:
        raise RuntimeError(
            "Projector received no input"
        )

    capture_input.append(
        inputs[0].detach().float().cpu()
    )


def post_hook(_module, _inputs, output):
    y = output

    if isinstance(y, (tuple, list)):
        if len(y) != 1:
            raise RuntimeError(
                "Unexpected projector tuple output"
            )
        y = y[0]

    if not torch.is_tensor(y):
        raise RuntimeError(
            "Projector output is not tensor"
        )

    capture_output.append(
        y.detach().float().cpu()
    )


pre_handle = projector.register_forward_pre_hook(
    pre_hook
)

post_handle = projector.register_forward_hook(
    post_hook
)

try:
    with torch.no_grad():
        for position, row in enumerate(
            rows,
            start=1,
        ):
            capture_input.clear()
            capture_output.clear()

            audio_path = (
                ROOT
                / row["audio_path"]
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": str(audio_path),
                        },
                        {
                            "type": "text",
                            "text": REAL_INSTRUCTION,
                        },
                    ],
                }
            ]

            batch = (
                processor
                .apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            )

            batch = {
                key: (
                    value.to(DEVICE)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value
                in dict(batch).items()
            }

            model(
                **batch,
                use_cache=False,
            )

            if len(capture_input) != 1:
                raise RuntimeError(
                    f"{row['example_id']}: "
                    f"projector input captures="
                    f"{len(capture_input)}"
                )

            if len(capture_output) != 1:
                raise RuntimeError(
                    f"{row['example_id']}: "
                    f"projector output captures="
                    f"{len(capture_output)}"
                )

            x = capture_input[0]
            y = capture_output[0]

            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]

            if y.ndim == 3 and y.shape[0] == 1:
                y = y[0]

            real_inputs.append(
                x
            )
            real_outputs.append(
                y
            )

            if (
                position == 1
                or position % 10 == 0
                or position == len(rows)
            ):
                print(
                    f"Real {position}/200 "
                    f"{row['split']} "
                    f"{row['condition']} "
                    f"in={tuple(x.shape)} "
                    f"out={tuple(y.shape)}",
                    flush=True,
                )

finally:
    pre_handle.remove()
    post_handle.remove()


# ------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------

synthetic_input_dim = (
    synthetic_inputs[0].shape[-1]
)
real_input_dim = (
    real_inputs[0].shape[-1]
)

synthetic_output_dim = (
    synthetic_outputs[0].shape[-1]
)
real_output_dim = (
    real_outputs[0].shape[-1]
)

if synthetic_input_dim != real_input_dim:
    raise RuntimeError(
        "Projector input feature mismatch: "
        f"synthetic={synthetic_input_dim} "
        f"real={real_input_dim}"
    )

if synthetic_output_dim != real_output_dim:
    raise RuntimeError(
        "Projector output feature mismatch: "
        f"synthetic={synthetic_output_dim} "
        f"real={real_output_dim}"
    )

print()
print("===== SHAPES =====")
print(
    "synthetic input unique:",
    sorted(
        {
            tuple(x.shape)
            for x in synthetic_inputs
        }
    ),
)
print(
    "real input unique:",
    sorted(
        {
            tuple(x.shape)
            for x in real_inputs
        }
    ),
)
print(
    "synthetic output unique:",
    sorted(
        {
            tuple(x.shape)
            for x in synthetic_outputs
        }
    ),
)
print(
    "real output unique:",
    sorted(
        {
            tuple(x.shape)
            for x in real_outputs
        }
    ),
)


results = {
    "synthetic_indices":
        synthetic_indices,
    "synthetic_n":
        len(synthetic_inputs),
    "real_n":
        len(real_inputs),
    "input": compare_space(
        "PROJECTOR INPUT",
        synthetic_inputs,
        real_inputs,
    ),
    "output": compare_space(
        "PROJECTOR OUTPUT",
        synthetic_outputs,
        real_outputs,
    ),
}


with (
    OUT / "comparison.json"
).open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )

np.save(
    OUT / "synthetic_input_pooled.npy",
    np.stack(
        [
            pool(x).numpy()
            for x in synthetic_inputs
        ]
    ),
)

np.save(
    OUT / "real_input_pooled.npy",
    np.stack(
        [
            pool(x).numpy()
            for x in real_inputs
        ]
    ),
)

np.save(
    OUT / "synthetic_output_pooled.npy",
    np.stack(
        [
            pool(x).numpy()
            for x in synthetic_outputs
        ]
    ),
)

np.save(
    OUT / "real_output_pooled.npy",
    np.stack(
        [
            pool(x).numpy()
            for x in real_outputs
        ]
    ),
)

print()
print(
    "Saved:",
    OUT / "comparison.json",
)
