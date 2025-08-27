# Bengali Figures of Speech — BengFoS 📖

[![paper badge](https://img.shields.io/badge/Paper-PDF-blue)](#paper) [![license](https://img.shields.io/badge/License-CC--BY--NC--SA-lightgrey)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-green)]() [![status](https://img.shields.io/badge/status-experimental-orange)]()

---

## 🔎 Project Snapshot

**Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification** — this repo contains the **BengFoS** dataset, code to reproduce the experiments (fine-tuning, quantized deployment, probing), visualizations, and the paper PDF in `/paper`. The README below includes three *featured figures* from the experiments placed for maximum visual appeal and quick insight.

---

# Highlights ✨

* **BengFoS Dataset** — Gold-standard, sentence-level annotations across several canonical Bengali poets; \~3,148 annotated sentences used for FoS experiments.
* **FoS Labels** - A complete list of Bengali figures of speech labels used for literary pieces' annotations.
  
<!-- Centered HTML table (recommended for GitHub README) -->
<table align="center" style="border-collapse:collapse; margin:0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; max-width:720px;">
  <thead>
    <tr>
      <th style="padding:6px 10px; text-align:center; border-bottom:1px solid #ddd;">SL. No.</th>
      <th style="padding:6px 10px; text-align:left; border-bottom:1px solid #ddd;">Figure of Speech</th>
      <th style="padding:6px 10px; text-align:center; border-bottom:1px solid #ddd;">Class Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:6px 10px; text-align:center;">1</td>
      <td style="padding:6px 10px; text-align:left;">None</td>
      <td style="padding:6px 10px; text-align:center;">0</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">2</td>
      <td style="padding:6px 10px; text-align:left;">Simile (উপমা)</td>
      <td style="padding:6px 10px; text-align:center;">1</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">3</td>
      <td style="padding:6px 10px; text-align:left;">Metaphor (রূপক)</td>
      <td style="padding:6px 10px; text-align:center;">2</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">4</td>
      <td style="padding:6px 10px; text-align:left;">Personification (মানবীকরণ)</td>
      <td style="padding:6px 10px; text-align:center;">3</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">5</td>
      <td style="padding:6px 10px; text-align:left;">Onomatopoeia (অনুকরণধ্বনি)</td>
      <td style="padding:6px 10px; text-align:center;">4</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">6</td>
      <td style="padding:6px 10px; text-align:left;">Hyperbole (অতিশয়োক্তি)</td>
      <td style="padding:6px 10px; text-align:center;">5</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">7</td>
      <td style="padding:6px 10px; text-align:left;">Alliteration (অনুপ্রাস)</td>
      <td style="padding:6px 10px; text-align:center;">6</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">8</td>
      <td style="padding:6px 10px; text-align:left;">Oxymoron and Antithesis, Epigram (বিরোধাভাস)</td>
      <td style="padding:6px 10px; text-align:center;">7</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">9</td>
      <td style="padding:6px 10px; text-align:left;">Irony (বিদ্রূপ)</td>
      <td style="padding:6px 10px; text-align:center;">8</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">10</td>
      <td style="padding:6px 10px; text-align:left;">Euphemism / Pun (শ্লেষ and যমক)</td>
      <td style="padding:6px 10px; text-align:center;">9</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">11</td>
      <td style="padding:6px 10px; text-align:left;">Apostrophe (উদ্দেশ্যোক্তি)</td>
      <td style="padding:6px 10px; text-align:center;">10</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">12</td>
      <td style="padding:6px 10px; text-align:left;">Synecdoche and Metonymy (প্রতিনিধিত্ব)</td>
      <td style="padding:6px 10px; text-align:center;">11</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; text-align:center;">13</td>
      <td style="padding:6px 10px; text-align:left;">Assonance (স্বরবৈশিষ্ট্য)</td>
      <td style="padding:6px 10px; text-align:center;">12</td>
    </tr>
  </tbody>
</table>

* **End-to-End Experiments** — Zero-shot baselines, fine-tuning (LoRA & full), quantized deployments, and layer-wise probing.
* **Featured Figures** — Three visually compelling figures (layer-wise probe, confusion matrix, token attention heatmap) are embedded below so visitors immediately see the research’s most important results.

---

# Featured figures — quick visual summary

> These figures are placed near the top of the README so readers instantly grasp the paper’s core findings. The images are referenced from the repository’s `Visualizations/` folder (these are the exact filenames from the Experiment.zip you provided).

<p align="center">
  <img src="Visualizations/Probing%20Visualizations/Layer%20Probing%20Results%20for%20DeepSeek%20R1%20Distilled%207B.png" alt="Layer-wise probing micro-F1 per layer" style="max-width:700px; width:100%; height:auto;">
</p>
**Figure 1 — Layer-wise probing (per-layer micro-F1).**  
*Short caption:* Probing logistic classifier micro-F1 for each model layer. FoS information peaks in mid-to-late layers (example: DeepSeek peak around mid layers). Place this directly under the Highlights so readers see where FoS cues are encoded.

---

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="Visualizations/Deployment%20Visualizations/fine_tuned_deepseek_7B_16-bit_deployment/DeepSeek_16-bit_Confusion%20Matrix.png"><img src="Visualizations/Deployment%20Visualizations/fine_tuned_deepseek_7B_16-bit_deployment/DeepSeek_16-bit_Confusion%20Matrix.png" alt="Confusion matrix" style="max-width:320px; width:100%; height:auto;"></a><br>
        **Figure 2 — Confusion matrix (test).**<br>
        *Short caption:* Per-label errors: common confusions (e.g., Metaphor ↔ Simile).
      </td>
      <td align="center">
        <a href="Visualizations/Probing%20Visualizations/DeepSeek%20R1%207B%20Attention%20Heatmaps/DeepSeek%20R1%20Sentence%20449.png"><img src="Visualizations/Probing%20Visualizations/DeepSeek%20R1%207B%20Attention%20Heatmaps/DeepSeek%20R1%20Sentence%20449.png" alt="Token attention heatmap" style="max-width:320px; width:100%; height:auto;"></a><br>
        **Figure 3 — Token-level attention heatmap.**<br>
        *Short caption:* Attention highlights tokens contributing to FoS decisions (useful for interpretability).
      </td>
    </tr>
  </table>
</div>

> Suggested placement: keep **Figure 1** under *Highlights* (full-width) and **Figures 2 & 3** side-by-side in the *Results* section (thumbnail row linking to full-size images in `Visualizations/Deployment Visualizations/...` and `Visualizations/Probing Visualizations/...`).

---

# 🗂 Repository layout (recommended)

```
├── README.md
├── LICENSE
├── paper/
│   └── Bengali_FoS_Identification.pdf
├── Visualizations/                # visualization outputs (as in Experiment.zip)
├── data/
├── src/
├── notebooks/
├── experiments/
├── results/
└── requirements.txt
```

---

## 🚀 Quickstart — run the experiments

### 1) Create environment & install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2) Prepare data (tokenize, clean, create splits)

```bash
# tokenizes, normalizes, and creates train/dev/test splits (example)
python src/utils.py \
  --prepare-data \
  --input data/raw \
  --output data/processed \
  --seq-length 128 \
  --seed 42
```

### 3) Optional — Upsampling / Augmentation (class balance)

Use upsampling when some FoS classes are underrepresented. This example creates an upsampled training file with a simple class-balance strategy.

```bash
# simple upsampling script (creates train_upsampled.jsonl)
python src/upsample.py \
  --input data/processed/train.jsonl \
  --output data/processed/train_upsampled.jsonl \
  --strategy class_balance \
  --target-min-count 500
```

Notes:

* `--strategy` can be `class_balance`, `syn_aug` (synonym substitution), or `backtranslate` if supported.
* Check `data/processed/` to confirm class distribution after upsampling.

### 4) Train

**LoRA (recommended for limited GPU memory)**:

```bash
python src/train.py \
  --model deepseek-r1-7b \
  --train data/processed/train_upsampled.jsonl \
  --dev data/processed/dev.jsonl \
  --output_dir models/deepseek-lora \
  --method lora \
  --lora_rank 8 \
  --batch_size 16 \
  --lr 3e-5 \
  --epochs 6 \
  --max_seq_length 128 \
  --fp16
```

**Full fine-tuning (if you have enough resources)**:

```bash
python src/train.py \
  --model deepseek-r1-7b \
  --train data/processed/train_upsampled.jsonl \
  --dev data/processed/dev.jsonl \
  --output_dir models/deepseek-ft \
  --method full \
  --batch_size 8 \
  --lr 2e-5 \
  --epochs 6 \
  --max_seq_length 128 \
  --fp16
```

**5-fold cross-validation (example using a config file)**:

```bash
python src/train.py --config experiments/cv_run.yaml
```

Tip: `experiments/cv_run.yaml` should define the model, data paths, and CV folds (the repo includes a template).

### 5) Evaluate (generate metrics & predictions)

```bash
python src/evaluate.py \
  --model_dir models/deepseek-lora \
  --test data/processed/test.jsonl \
  --out results/deepseek-lora-eval.json \
  --metrics micro_f1 macro_f1 per_label
```

Output: a JSON (or CSV) with per-label precision/recall/F1 and overall metrics.

### 6) Quantize & deploy (16-bit / 8-bit options)

**Quick quantize + eval (example with evaluate.py quantize flag)**:

```bash
# quantize in-place (or into a new dir) and evaluate quantized model
python src/evaluate.py \
  --model_dir models/deepseek-lora \
  --quantize 16 \
  --quant_out models/deepseek-lora-16bit \
  --test data/processed/test.jsonl \
  --out results/deepseek-lora-16bit-eval.json
```

Or use a separate `src/quantize.py` if present:

```bash
python src/quantize.py --input models/deepseek-lora --bits 16 --output models/deepseek-lora-16bit
python src/evaluate.py --model_dir models/deepseek-lora-16bit --test data/processed/test.jsonl --out results/deepseek-lora-16bit-eval.json
```

Tip: For production with constrained memory, **LoRA + 16-bit quantization** is a practical trade-off.

### 7) Probing & interpretability (layer probes, attention maps)

Run the layer-wise probing pipeline to locate where FoS information is encoded and to produce probe outputs for plotting:

```bash
python src/probe.py \
  --model_dir models/deepseek-lora \
  --data data/processed/test.jsonl \
  --layers all \
  --probe_out results/probes/ \
  --save_token_attentions Visualizations/Probing\ Visualizations/Attention_Examples/
```

What this does (expected outputs):

* Trains logistic probes on hidden states for each layer and saves per-layer micro-F1 scores in `results/probes/`.
* Dumps attention maps (or selected example heatmaps) to `Visualizations/Probing Visualizations/`.
* Optionally saves CSVs with probe predictions per-layer for deeper error analysis.

### 8) Helpful utilities

* Produce plots from saved results (if `plot_utils.py` exists):

```bash
python src/plot_utils.py --probe_results results/probes/ --eval results/deepseek-lora-eval.json --out Visualizations/
```

* Inspect per-label errors or generate a confusion matrix:

```bash
python src/analysis_utils.py --pred results/deepseek-lora-eval.json --confusion results/confusion_deepseek.png
```

---

### Quick Troubleshooting

* If GPU memory is limited: use `--method lora --fp16` and smaller `--batch_size`.
* Keep a `experiments/` YAML for reproducibility (seed, model, tokenizer, augmentation flags).
* Verify class balance after upsampling: `python -c "import json,collections; print(collections.Counter([json.loads(l)['label'] for l in open('data/processed/train_upsampled.jsonl')]))"`
* All scripts accept `--seed` for reproducibility; use the same seed across prep, train, and probe runs.

---

# 📊 Results (with featured figures)

The featured figures summarize the paper’s takeaways:

* **Figure 1 (Layer probe)**: Shows where FoS signals are strongest in the network — helps justify probe and interpretability claims.
* **Figure 2 (Confusion matrix)**: Reveals label-pair weaknesses — useful for future annotation/augmentation.
* **Figure 3 (Attention heatmap)**: Makes the model’s focus interpretable at token level and is highly shareable in talks and the README.

Include a captioned table with the final numbers (replace with exact experiment outputs):

| Setup                           | Metric (micro-F1) |
| ------------------------------- | ----------------: |
| Zero-shot Llama-3 (8B)          |             0.418 |
| Fine-tuned Llama-3 (16-bit)     |              0.54 |
| Fine-tuned DeepSeek R1 (16-bit) |              0.56 |

---

# 🔬 Probing & interpretability

* `src/probe.py` runs layer-wise probes and saves `Visualizations/Probing Visualizations/Layer Probing Results for DeepSeek R1 Distilled 7B.png`.
* `notebooks/` includes interactive notebooks that reproduce Figures 2 & 3 from the paper.
* When generating attention heatmaps, choose short, high-confidence test examples for the clearest visualization.

---

# 🧾 Paper & Citation

Paper: **Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification** — PDF in `/paper`.

BibTeX:

```bibtex
@inproceedings{das2025bengfos,
  title = {Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification},
  author = {Sourav Das and Kripabandhu Ghosh},
  year = {2025},
  note = {BengFoS dataset + experiments; PDF in /paper}
}
```

---
