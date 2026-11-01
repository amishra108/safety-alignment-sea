# Safety Alignment of Audio-Language Models: SEA Replication

This repository contains experiments adapting the methodology from the SEA (Safety Embedding Alignment) paper to additional audio-language models.

The project investigates whether SEA-style alignment generalises across model architectures and evaluation benchmarks, comparing baseline models with supervised fine-tuning (SFT) and direct preference optimisation (DPO).

## Models

Three audio-language model families are investigated:

- Qwen2.5-Omni-7B
- SALMONN-2
- Video-SALMONN-2 (4B)

The experiments include dataset preparation, synthetic audio embedding optimisation, safety training, evaluation, and post-training parameter analysis.

Qwen3-4B and other auxiliary models support translation, text preparation, and response adjudication. Speech generation is performed using dedicated text-to-speech models.

## Training

The alignment pipeline follows the SEA approach, using safety examples derived from PKU-SafeRLHF.

The prepared dataset contains 3,000 examples, consisting of 2,000 harmful and 1,000 harmless examples.

The experiments investigate synthetic audio embedding optimisation, supervised fine-tuning, direct preference optimisation, and the differences between trained and baseline models.

## Evaluation Benchmarks

### Multilingual Audio Jailbreak Clean Benchmark

This benchmark was developed in the earlier [safety-alignment project](https://github.com/amishra108/safety-alignment).

It extends 100 harmful and 100 benign requests across four languages (English, Chinese, Russian, and Spanish), four prompting conditions (original, literal, narrative framing, and content dilution), and two modalities (text and synthesised audio).

This produces **6,400 evaluation inputs**.

The Chinese, Russian, and Spanish translations and corresponding audio were created through the previous project's preparation pipeline. This repository includes the benchmark manifests rather than the full audio collection.

### VA-SafetyBench

VA-SafetyBench is used to evaluate harmful compliance across text-only, audio-only, and KPT conditions, including category-level attack success rates.

### AIRBench

AIRBench is used for additional audio-language evaluation, including safety-related performance comparisons between baseline and aligned models.

## Post-Training Analysis

The project investigates how SFT and DPO affect model parameters.

Analyses include relative weight changes, cosine similarity, parameter-update directions, and layer-wise comparisons of decoder or Thinker projections.

## Repository Structure

- `scripts/` — Dataset preparation, embedding optimisation, training, evaluation, and diagnostics.
- `data/eval/` — Evaluation data.
- `data/saferlhf/` — Prepared safety-alignment training data.
- `data/audio_jailbreak_clean_benchmark/manifests/` — Multilingual benchmark manifests.
- `data/va_safety_bench/processed/` — Processed VA-SafetyBench manifests.
- `data/airbench/processed/` — Processed AIRBench data.
- `models/` — Descriptions of base models and trained checkpoint families.
- `outputs/` — Selected experimental outputs.
- `results/` — Evaluation results and analysis artefacts.

## Using the Codebase

The repository consists of research scripts rather than a standalone Python package.

Obtain the pretrained models from their original sources, prepare the necessary datasets, and run the relevant numbered scripts under `scripts/`.

The code was developed for GPU-enabled research environments. Model paths, dependencies, hardware configurations, and checkpoint locations may need to be adjusted.

Large model weights, checkpoints, raw audio collections, external repositories, and logs are not distributed.
