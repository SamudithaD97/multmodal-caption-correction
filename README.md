# CaptionFixer-Transformer
<div align="center">

### Multimodal Caption Detection and Correction using Transformer-based Vision-Language Models

Detect incorrect image captions and automatically generate corrected captions using multimodal deep learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# Overview

Traditional image captioning systems focus on generating captions from images.  
However, real-world captions are often:

- Incorrect
- Incomplete
- Misleading
- Noisy

This project introduces a **multimodal transformer-based framework** that can:

1. Detect whether a caption matches an image
2. Generate a corrected caption when the caption is wrong

The system combines:
- **Discriminative learning** (classification)
- **Generative learning** (caption correction)

within a unified multimodal architecture.

---

# Key Features

✅ Caption Match / Mismatch Classification  
✅ Transformer-based Caption Correction  
✅ Cross-modal Attention Fusion  
✅ Vision Transformer (ViT) / ResNet Encoders  
✅ Caption Dropout Regularization  
✅ Token Saliency Analysis  
✅ Grad-CAM Visual Explanations  
✅ Multi-task Learning Framework  

---

# Model Architecture

## Transformer-Based Multimodal Model

The architecture contains four major components:

### 1. Image Encoder
Extracts visual embeddings from the input image.

Supported encoders:
- ResNet-50
- Vision Transformer (ViT)

---

### 2. Text Encoder
Encodes the input caption using a Transformer encoder.

Responsibilities:
- Token embedding
- Positional encoding
- Caption representation learning

---

### 3. Cross-Attention Fusion
Combines:
- Visual embeddings
- Text embeddings

This allows the model to learn semantic alignment between image and caption.

---

### 4. Transformer Decoder
Autoregressively generates corrected captions token-by-token.

Outputs:
- Corrected caption
- Vocabulary probability distribution

---

# Task Definition

The model performs two tasks simultaneously.

## 1. Caption Classification

Determine whether the caption matches the image.

### Output

```python
0 -> Incorrect Caption
1 -> Correct Caption
```

---

## 2. Caption Correction

Generate a corrected caption if the input caption is incorrect.

### Example

### Input

```text
"A traffic light near church at dusk."
```

### Output

```text
"some broccoli a piece of chicken and some white rice"
```

---

# Dataset

The project uses a modified COCO-based multimodal dataset.

Each sample contains:

| Field | Description |
|---|---|
| image | Input image |
| input_caption | Possibly incorrect caption |
| target_caption | Correct caption |
| label | Match / mismatch label |

---

## Dataset Example

```json
{
  "sample_id": "train-003016-neg",
  "source_image_id": 26981,
  "image_path": "coco-data/images/train2017/000000026981.jpg",
  "input_caption": "Street signal light near church at dusk near streetlight.",
  "target_caption": "some broccoli a piece of chicken and some white rice",
  "label": 0,
  "pair_type": "negative"
}
```

---

# Preprocessing

## Image Preprocessing

- Resize to `224 × 224`
- Normalize using ImageNet statistics

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

---

## Text Preprocessing

- Tokenization
- Attention masks
- Padding
- Special tokens

```python
[CLS] caption tokens [SEP]
```

---

# Training Strategy

The model uses a multi-task learning objective.

---

## 1. Discriminative Loss

Binary classification loss for caption matching.

```math
L_{match} =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log(\hat{y_i})
+
(1-y_i)\log(1-\hat{y_i})
\right]
```

---

## 2. Generative Loss

Autoregressive caption generation loss.

```math
L_{caption} =
-\frac{1}{T}
\sum_{t=1}^{T}
\log P(w_t | w_{<t}, v_{img}, v^{masked}_{text})
```

---

## 3. Total Loss

```math
L_{total} = L_{match} + \lambda L_{caption}
```

---

# Caption Dropout

## Problem — "Copycat Bug"

The decoder initially learned to simply copy the input caption instead of correcting it.

---

## Solution — Caption Dropout

Randomly mask input caption tokens during training.

### Benefits
- Forces visual grounding
- Reduces language shortcut learning
- Improves caption correction quality
- Increases robustness

---

# Interpretability

---

## Token Saliency

Measures token importance using occlusion.

### Formula

```math
Saliency = P_{original} - P_{masked}
```

### Observation
The model strongly focuses on:
- Important nouns
- Verbs
- Semantic keywords

while ignoring:
- Function words
- Grammar-only tokens

---

## Grad-CAM Visualization

Grad-CAM is used to visualize image regions influencing prediction decisions.

### Findings
- The model attends to semantically meaningful image regions
- Visual grounding is successfully achieved

---

# Results

# Classification Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.8315 |
| Precision | 0.7995 |
| Recall | 0.8850 |
| F1 Score | 0.8401 |
| Validation Loss | 0.3908 |

---

# Generation Performance

| Metric | Value |
|---|---:|
| Caption Loss | 2.9223 |
| BLEU-1 | 0.4133 |
| BLEU-2 | 0.2781 |
| BLEU-3 | 0.1939 |
| BLEU-4 | 0.1413 |
| ROUGE-L | 0.4128 |

---

# Hyperparameter Tuning

## Search Space

| Parameter | Values |
|---|---|
| Learning Rate | 1e-4, 2e-4 |
| Caption Dropout | 0.2 - 0.4 |
| λ_caption | 1.0 - 2.0 |
| d_model | 256, 384 |
| Dropout | 0.1, 0.2 |

---

## Best Configuration

```yaml
learning_rate: 2e-4
caption_dropout_prob: 0.3
lambda_caption: 1.5
d_model: 256
dropout: 0.1
```

---

# Repository Structure

```bash
CaptionFixer-Transformer/
│
├── data/
│   └── coco_correction/
│
├── logs/
├── results/
│
├── scripts/
│   ├── build_coco_pairs.py
│   ├── build_coco_pairs_download.py
│   └── run_sbatch_job.sh
│
├── src/
│   ├── collate.py
│   ├── dataloaders.py
│   ├── dataset.py
│   ├── gradcam.py
│   ├── infer.py
│   ├── model.py
│   ├── saliency.py
│   ├── train.py
│   ├── test.py
│   └── test_dataset_with_batching.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# How to Run

## 1. Create Environment

```bash
conda create -n dl_env python=3.10
conda activate dl_env
```

---

## 2. Install Dependencies

```bash
pip install torch torchvision transformers pycocotools pillow
```

---

## 3. Build COCO Dataset

```bash
mkdir -p coco/images coco/annotations

cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip -d images/
unzip annotations_trainval2017.zip

cd ..
```

---

## 4. Build Dataset (Generate Caption Pairs)

> Skip this step if the generated `.jsonl` manifests are already included in the repository.

Keep the image locations exactly as referenced in the manifests:

```text
coco-data/images/train2017/000000237912.jpg
```

Your downloaded COCO dataset should exist in the project root directory.

### Generate pairs manually

```bash
python scripts/build_coco_pairs.py \
  --ann-file coco/annotations/captions_train2017.json \
  --images-root coco/images/train2017 \
  --out-dir data/coco_correction \
  --num-source-images 10000
```

---

## 5. Test the Dataloader

```bash
python src/test_dataset_with_batching.py
```

At this stage:
- False and true captions are generated
- Images are loaded and normalized
- Captions are tokenized
- Training batches are prepared

The processed dataset is now ready to be fed into the multimodal transformer model.

---

# Training / Testing / Inference

## Run on IU Quartz Cluster

Submit the training job:

```bash
sbatch ./scripts/run_sbatch_job.sh
```

This schedules the job and returns a job ID.

---

## Select Task

Inside `run_sbatch_job.sh`, uncomment the appropriate line for:

- Training
- Testing
- Inference
- Saliency visualization
- Grad-CAM visualization

---

## Stream Logs

Logs are written to:

```bash
./logs/job_{job_id}.out
```

Monitor logs live:

```bash
tail -f ./logs/job_{job_id}.out
```

---

# Example Output

## Input Caption

```text
"A man walking near a traffic light."
```

## Predicted Match Probability

```text
0.12
```

## Corrected Caption

```text
"A plate with broccoli, chicken, and white rice."
```

---

# Experimental Findings

## ResNet vs ViT

| Aspect | ResNet | ViT |
|---|---|---|
| Accuracy | Higher | Slightly Lower |
| Convergence | Faster | Slower |
| Interpretability | Smooth Heatmaps | Sharp Attention Maps |
| Data Efficiency | Better | Lower |

---

# Problems Encountered

- Initial dataset size was extremely large (~90GB)
- Transformer training complexity
- Caption copying issue
- Limited training epochs
- Evaluation instability

---

# Future Work

## Planned Improvements

- Train on larger real-world datasets:
  - NewsCLIPpings
  - VisualNews

- Use patch-level image tokens
- Improve decoder architecture
- Better multimodal fusion strategies
- Use pretrained BERT embeddings
- Improve evaluation metrics
- Add beam search decoding
- Add instruction tuning

---

# Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| NLP | HuggingFace Transformers |
| Vision | torchvision |
| Visualization | Grad-CAM |
| Dataset | COCO |
| Training | Multi-task Learning |

---

# References

```bibtex
@article{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford et al.},
  year={2021}
}

@article{vinyals2015show,
  title={Show and Tell: A Neural Image Caption Generator},
  author={Vinyals et al.},
  year={2015}
}

@article{lu2019vilbert,
  title={ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations},
  author={Lu et al.},
  year={2019}
}

@article{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training},
  author={Li et al.},
  year={2022}
}
```

---

# Authors

- Kushal Pokharel
- Hamza Almani
- Samuditha Wijayasundara

---

# License

MIT License
