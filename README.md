<div style="display: flex; justify-content: center; margin-bottom: 10px;">
    <img src="Framework_Architecture.png" style="width: 100vw; height: auto;">
</div>

<a href="#">
<h1 align="left">
   🩺 MedMask: A Self-Supervised Masking Framework for Medical Image Detection Using VFMs
</h1>
</a>

# 🎯 What is MedMask?

MedMask is a self-supervised masking framework designed to enhance medical image detection, particularly for breast cancer detection from mammograms (BCDM). It addresses the challenge of limited annotated datasets by leveraging masked autoencoders (MAE) and vision foundation models (VFMs) within a transformer-based architecture. The framework employs a customized MAE module that masks and reconstructs multi-scale feature maps, allowing the model to learn domain-specific characteristics more effectively. Additionally, MedMask integrates an expert contrastive knowledge distillation technique, utilizing the zero-shot capabilities of VFMs to improve feature representations. By combining self-supervised learning with knowledge distillation, MedMask achieves state-of-the-art performance on publicly available mammogram datasets like INBreast and DDSM, demonstrating significant improvements in sensitivity. Its applicability extends beyond medical imaging, showcasing generalizability to natural image tasks.

# ⚙️ Pipeline

Our method involves two main stages:

## 1. Extract Embeddings Using BiomedParse

We utilize BiomedParse, a powerful Vision Foundation Model (VFM) introduced in Nature Methods, to generate rich, task-agnostic embeddings from biomedical images, including mammograms. Trained across nine diverse modalities, BiomedParse performs joint segmentation, detection, and recognition using its unified architecture and pre-trained checkpoints. These robust embeddings serve as the foundational input for our MedMask framework, enabling superior generalization and domain understanding even with limited annotations.

<b>🔗 BiomedParse Links</b>
<a href="https://www.nature.com/articles/s41592-024-02499-w.epdf?sharing_token=13MXNEG8f5JOr30TorWJXNRgN0jAjWel9jnR3ZoTv0PZugrLJtTZcQj4PJckxX_PaGqvO3y6jPvaAlFxqlfW8F1tqAukjnPV-aqr4s4izWg_qebtOm7qbi6Z08lkjqQOOaSe7JB9tCb23TCY3OrElyRUhgGtiROQd3xy4AwyZIsjw-5m2Cx8bag044uNrqQHwSazsFcyeiEYlaP6lqewuQu0xEd5yA0CQhB-2umuHSM%3D">📄 Paper</a> | <a href="https://github.com/microsoft/BiomedParse">🏥 Repository</a>

### 🔧 Installation

```bash
cd BiomedParse
conda env create -f environment.yml
conda activate biomedparse
pip install -r assets/requirements/requirements.txt
```

### Install PyTorch (adjust CUDA version as per your system):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 🧠 Inference - To generate embeddings on mammogram datasets:

#### 1. Update path in generate_embedings.py file.

```python
input_dir = '/home/Drive/Datasets/BCD_INBreast/coco_uniform_ts/train2017'
embeddings_dir = '/home/Drive/Outputs/embeddings/train_inbreast'
prompts = ['malignant cancer in the breast']
```

#### 2. run generate_embedings.py

```bash
python generate_embedings.py
```

Model checkpoints are auto-loaded from HuggingFace.

## 2. Train MedMask with Self-Supervised Masking + VFMs

After extracting embeddings, MedMask takes over. It leverages:

- A Masked Autoencoder (MAE) on multi-scale features.
- Expert-Guided Contrastive Distillation from VFM.
- A Transformer-based backbone for detection.
- Trained on datasets like INBreast, DDSM, and RSNA-BSD1K.

### 📂 Datasets

We evaluate MedMask on the following mammogram datasets:
| Dataset | Samples | Malignant Cases | Format |
| ---------- | ------- | --------------- | -------------- |
| DDSM | 1324 | 573 | COCO-JSON |
| INBreast | 200 | 41 | COCO-JSON |
| RSNA-BSD1K | 1000 | 200 | COCO-JSON |

### What is RSNA-BSD1K Data?

RSNA-BSD1K is a bounding box annotated subset of 1,000 mammograms from the RSNA Breast Screening Dataset, designed to support further research in breast cancer detection from mammograms (BCDM). The original RSNA dataset consists of 54,706 screening mammograms, containing 1,000 malignancies from 8,000 patients. From this, we curated RSNA-BSD1K, which includes 1,000 mammograms with 200 malignant cases, annotated at the bounding box level by two expert radiologists.

```bash
[data_root]
└─ inbreast/
   └─ annotations/
   └─ images/train/, val/, test/
└─ ddsm/
   └─ annotations/
   └─ images/train/, val/, test/
└─ rsna-bsd1k/
   └─ annotations/
   └─ images/train/, val/, test/
```

### 🔧 Installation

#### 1. Requirements

- Linux, CUDA >= 11.1, GCC >= 8.4

- Python >= 3.8

- torch >= 1.10.1, torchvision >= 0.11.2

- Other requirements

  ```bash
  pip install -r requirements.txt
  ```

#### 2. Compiling Deformable DETR CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### 📊 Data preparation

We provide the three benchmark datasets used in our experiments:

- **BCD_DDSM**: The DDSM dataset consists of 1,324 mammography images, including 573 malignant cases.
- **BCD_InBreast**: The INBreast dataset contains 200 images from 115 patients, with 41 malignant and 159 benign cases.
- **BCD_RSNA**: The RSNA-BSD1K dataset is a curated subset of 1,000 mammograms from the original RSNA dataset, including 200 malignant cases with bounding box annotations.

You can download the raw data from the official websites: [dataset]() and organize the datasets and annotations as follows:

```bash
[data_root]
└─ inbreast
	└─ annotations
		└─ instances_train.json
		└─ instances_val.json
	└─ images
		└─ train
		└─ val
└─ ddsm
	└─ annotations
		└─ instances_train.json
		└─ instances_val.json

	└─ images
		└─ train
		└─ val
└─ rsna-bsd1k
	└─ annotations
		└─ instances_full.json
		└─ instances_val.json
	└─ images
		└─ train
		└─ val
```

To use additional datasets, you can edit [datasets/coco_style_dataset.py](https://github.com/JeremyZhao1998/MRT-release/blob/main/datasets/coco_style_dataset.py) and add key-value pairs to `CocoStyleDataset.img_dirs` and `CocoStyleDataset.anno_files` .

### 🚀 Training and Evaluation

Our method follows a three-stage training paradigm to optimize efficiency and performance. The initial stage involves standard supervised training using annotated mammograms. The second stage incorporates self-supervised masked autoencoding (MAE) to refine representations from unannotated data. The final stage introduces Expert-Guided Fine-Tuning, leveraging vision foundation models (VFMs) for enhanced feature learning and generalization.

For training on the DDSM-to-INBreast benchmark, update the files in `configs/def-detr-base/ddsm/` to specify `DATA_ROOT` and `OUTPUT_DIR`, `embeddings_dir` then execute:

```sh
sh configs/def-detr-base/ddsm/source_only.sh      # Stage 1: Baseline Object Detector Training
sh configs/def-detr-base/ddsm/cross_domain.sh     # Stage 2: Masked Autoencoder Training
```

## 🧪 Evaluation & Results

MedMask achieves state-of-the-art results across mammogram datasets with significant gains in sensitivity and cross-domain generalization, particularly in the low-data regime.

#### 📌 Note: Use ` sh configs/def-detr-base/froc_predictions.sh` for evaluation on test sets.

## 📈 Model Checkpoints

We conduct all experiments with batch size 8 (for source_only stage, 8 labeled samples; for cross_domain_mae and MRT teaching stage, 8 labeled samples and 8 unlabeled samples), on 4 NVIDIA A100 GPUs.

| Dataset    | Encoder Layer | Decoder Layer | R@0.3 | Weights                                                                                          |
| ---------- | ------------- | ------------- | ----- | ------------------------------------------------------------------------------------------------ |
| RSNA-BSD1K | 6             | 6             | 0.886 | [Download](https://drive.google.com/drive/folders/1utNXOqhsSTscPfrbwwfyZI-f9m7FCTSh?usp=sharing) |
| DDSM       | 6             | 6             | 0.718 | [Download](https://drive.google.com/drive/folders/1utNXOqhsSTscPfrbwwfyZI-f9m7FCTSh?usp=sharing) |
| INBREAST   | 6             | 6             | 0.888 | [Download](https://drive.google.com/drive/folders/1utNXOqhsSTscPfrbwwfyZI-f9m7FCTSh?usp=sharing) |

## 📝 Citation

```bibtex

```

## Acknowledgements

- BiomedParse team (Microsoft Research)
- DDSM, INBreast, and RSNA dataset contributors
- Open-source VFM and MAE research communities
