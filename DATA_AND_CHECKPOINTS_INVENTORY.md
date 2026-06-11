# PCR Simulation: Data Files and Model Checkpoints Inventory

## Executive Summary
This document catalogs all data files and model checkpoints referenced in the PCR simulation project, identifying which files are present locally and which need to be downloaded from the remote server (`/media/ssd1/huong/PCR-huong/data/`).

---

## Model Checkpoints

### **Present Locally**
| File | Location | Architecture | Notes |
|------|----------|--------------|-------|
| `fusiongene_large.ckpt` | `/model_weights/` | GeneFusionHeadsModel (PyTorch Lightning) | 1.09 GB - Main production model |

### **Referenced but NOT Present (Need to Download)**

#### From `/media/ssd1/huong/PCR-huong/data/model_checkpoints/`

| Checkpoint Path | Architecture | Format | Used In | Purpose |
|-----------------|--------------|--------|---------|---------|
| `fusion_model/best_model_v2.pth` | FusionModel (non-ViT) | PyTorch | `run_models.ipynb` | Early fusion model without ViT |
| `fusion_model/07_01_human_label_vit_fusioin_epoch=49-step=3050.ckpt` | ViTFusionModel | Lightning (loaded into non-Lightning) | `run_models.ipynb` | Human-labeled ViT fusion model |
| `fusion_model/10_27_fusion_model_vit_delta64.pth` | ViTFusionModel (delta=64) | PyTorch | `experimental_models.ipynb`, `run_models.ipynb` | Main ViT fusion model with delta features |
| `experimental/seq_best.ckpt` | SeqModel | PyTorch Lightning | `experimental_models.ipynb` | Sequence-only model |
| `experimental/seq_gene_best.ckpt` | SeqGeneModel | PyTorch Lightning | `experimental_models.ipynb` | Sequence + gene indicator model |
| `experimental/seq_curve_gene_best.ckpt` | SeqCurveGeneModel | PyTorch Lightning | `experimental_models.ipynb` | Sequence + curve + gene model |
| `experimental/epoch=37-step=2090.ckpt` | CurveShapeModel | PyTorch Lightning | `experimental_models.ipynb` | Image/curve shape model |

#### From `/media/ssd1/huong/PCR-huong/data/new_data/model_checkpoints/`

| Checkpoint Path | Architecture | Format | Used In | Purpose |
|-----------------|--------------|--------|---------|---------|
| `Seq_Model_large.ckpt` | SeqModel | PyTorch Lightning | `evaluation/new_data/model_evaluation.ipynb` | Large sequence model on new data |
| `Seq_Gene_Model_large.ckpt` | SeqGeneModel | PyTorch Lightning | Implied by naming pattern | Large sequence+gene model on new data |
| `Image_Model_large.ckpt` | CurveShapeModel | PyTorch Lightning | Implied by naming pattern | Large image model on new data |

#### Local Checkpoints (in `output/` directory - may be present)

| Checkpoint Path | Architecture | Format | Used In | Purpose |
|-----------------|--------------|--------|---------|---------|
| `output/fusion_model/10_31_ensemble_model_vit_delta64new.pth` | EnsembleModel wrapping ViTFusionModel | PyTorch | `run_models.ipynb` | Ensemble with IGI call integration |
| `output/fusion_model/fusion_w_gene.pth` | FusionwGeneModel | PyTorch | `run_models.ipynb` | Fusion with gene indicators |
| `output/fusion_model/fusion_w_no_pretain.pth` | FusionModel | PyTorch | `run_models.ipynb` | Fusion without pretraining |
| `output/image_model/image_best_model_ep50.pth` | ImageModel | PyTorch | `run_models.ipynb` | Image-only model (50 epochs) |
| `output/seq_model/seq_best_model_ep50.pth` | SequenceModel | PyTorch | `run_models.ipynb` | Sequence-only model (50 epochs) |
| `output/10_27_fusion_model_vit_delta64/best_model_no_pretrain.pth` | ViTFusionModel | PyTorch | Training scripts | ViT fusion model checkpoint |

---

## Model Architectures Summary

### **FusionModel / ViTFusionModel**
- **Input**: Image (224x224), Sequence (40 cycles), Gene indicator (6 genes one-hot)
- **Components**: 
  - ViT-B/32 (pretrained on ImageNet) for image processing
  - LSTM for sequence processing
  - LSTM for delta sequence (first differences)
  - Multi-head prediction (3 heads: groundtruth, IGI FP, IGI FN)
- **Parameters**: `hidden_size=512`, `latent_dim=512`, `num_layers=3`, `delta=64`, `num_heads=3`
- **Files**: `x02_train_fusion_ViT_img_seq_gene.py`, `x02_train_fusion_img_seq_gene.py`

### **EnsembleModel**
- **Input**: Wraps FusionModel + IGI call
- **Components**: FusionModel outputs + IGI call → Linear layer → Sigmoid
- **Files**: `x02_train_fusion_ViT_ensemble.py`

### **SeqModel / SeqGeneModel** (PyTorch Lightning)
- **Input**: Sequence (40 cycles), optionally Gene indicator
- **Components**: LSTM → FC layers → Multi-head prediction
- **Parameters**: `hidden_size=512`, `latent_dim=512`, `num_layers=5`
- **Files**: `deprecated/pytorch_lightning/src/pcr_lightning.py`

### **CurveShapeModel / ImageModel** (PyTorch Lightning)
- **Input**: Curve image (224x224)
- **Components**: ViT or ResNet → FC layers
- **Files**: `deprecated/pytorch_lightning/src/pcr_lightning.py`

### **GeneFusionHeadsModel** (PyTorch Lightning)
- **Input**: Image + Sequence + Gene indicator
- **Components**: Similar to ViTFusionModel but in Lightning framework
- **Current checkpoint**: `model_weights/fusiongene_large.ckpt`

---

## Data Files

### **Present Locally in `/data/`**

#### **Main Training/Evaluation Data**
| File | Size | Description |
|------|------|-------------|
| `groundtruth_df.csv` | 88.3 MB | Main ground truth labels |
| `groundtruth_df_curve_dict.pkl` | 10.2 MB | Curve data dictionary (old version) |
| `groundtruth_df_curve_dict_split_v2.pkl` | 9.9 MB | Curve data with train/val/test splits |
| `groundtruth_df_target_data.csv` | 2.2 MB | Target labels |
| `groundtruth_df_target_data_split_v2.csv` | 2.3 MB | Target labels with splits (v2) |

#### **New/Updated Data (no_invalid versions)**
| File | Size | Description |
|------|------|-------------|
| `new_groundtruth_df.csv` | 923.3 MB | New ground truth data |
| `new_groundtruth_df_no_invalid.csv` | 983.6 MB | New data excluding invalid samples |
| `new_groundtruth_df_curve_dict_fn.pkl` | 101.4 MB | Fluorescence (Fn) curves |
| `new_groundtruth_df_curve_dict_fn_no_invalid.pkl` | 111.6 MB | Fn curves (no invalid) |
| `new_groundtruth_df_curve_dict_drn.pkl` | 101.4 MB | Delta Rn (drn) curves |
| `new_groundtruth_df_curve_dict_drn_no_invalid.pkl` | 111.6 MB | drn curves (no invalid) |
| `new_groundtruth_df_target_data.csv` | 22.6 MB | New target labels |
| `new_groundtruth_df_target_data_no_invalid.csv` | 25.1 MB | New target labels (no invalid) |
| `new_full_curve_dict_fn.pkl` | 544.0 MB | Full Fn curve dictionary |
| `new_full_curve_dict_fn_no_invalid.pkl` | 544.0 MB | Full Fn curves (no invalid) |
| `new_full_curve_dict_drn.pkl` | 544.0 MB | Full drn curve dictionary |
| `new_full_curve_dict_drn_no_invalid.pkl` | 544.0 MB | Full drn curves (no invalid) |

#### **Retest Data**
| File | Size | Description |
|------|------|-------------|
| `new_retest_curve_dict.pkl` | 7.4 MB | Retest curve data |
| `new_retest_curve_dict_1.pkl` | 10.6 MB | Retest curves (version 1) |
| `new_retest_df_target_data.csv` | 4.9 MB | Retest target labels |
| `new_retest_df_target_data_1.csv` | 7.3 MB | Retest labels (version 1) |
| `new_retested_curve_dict.pkl` | 5.7 MB | Retested sample curves |
| `new_retested_df_target_data.csv` | 3.5 MB | Retested sample labels |

#### **External Validation Datasets**
| File | Size | Description |
|------|------|-------------|
| `chip60_curve_dict.pkl` | 13.5 KB | ChipPCR C60 amplification data |
| `chip60_target_data.csv` | 2.0 KB | ChipPCR labels |
| `karlen_curve_dict.pkl` | 104.0 KB | Karlen et al. qPCR data |
| `karlen_target_data.csv` | 20.7 KB | Karlen labels |
| `known_curve_dict.pkl` | 121.2 KB | Known dilution curves |
| `known_target_data.csv` | 42.3 KB | Known dilution labels |

#### **Model Outputs (Present)**
Located in `/data/model_outputs/`:
- `3_20_ImageModel_large_no_invalid_*_val_test_pred_df.csv` (5 files)
- `3_20_SeqModel_large_no_invalid_*_val_test_pred_df.csv` (5 files)
- `3_20_SeqGeneModel_large_no_invalid_*_val_test_pred_df.csv` (5 files)
- `4_3_fusiongene_large_no_invalid_*_val_test_pred_df.csv` (6 files)

### **Referenced but NOT Present (Need to Download)**

#### From `/media/ssd1/huong/PCR-huong/data/`

**Core Data Files:**
| File | Description | Used In |
|------|-------------|---------|
| `data.h5` | HDF5 file with keys: `curve_data`, `sample_info`, `igi_gene_call` | Multiple evaluation notebooks |
| `retest_df.csv` | Retest samples dataframe | `evaluation/model_eval_retest.ipynb` |
| `chipPCR_C60amp.csv` | ChipPCR C60 amplification raw data | `run_models.ipynb` |
| `qpcr_karlen.csv` | Karlen qPCR raw data | `run_models.ipynb` |
| `known_dilution_data.csv` | Known dilution raw data | `run_models.ipynb` |
| `new_join_df.pkl` | Joined dataframe (new data) | `evaluation/paper_plots/paper_plots_retest.ipynb` |

**Model Output Files:**
| File | Description |
|------|-------------|
| `model_outputs/fusion_vit_delta64_test_pred_df.csv` | Test predictions from ViT fusion model |
| `model_outputs/fusion_vit_delta64_val_pred_df.csv` | Validation predictions |
| `model_outputs/fusion_vit_delta64_train_pred_df.csv` | Training predictions |
| `model_outputs/fusion_vit_delta64_retest_pred_df.csv` | Retest predictions |
| `model_outputs/fusion_vit_delta64_clinical_pred_df.csv` | Clinical sample predictions |
| `model_outputs/fusion_vit_delta64_chip60_pred_df.csv` | ChipPCR predictions |
| `model_outputs/fusion_vit_delta64_karlen_pred_df.csv` | Karlen predictions |
| `model_outputs/fusion_vit_delta64_known_pred_df.csv` | Known dilution predictions |
| `model_outputs/human_based_fusion_vit_delta64_*_pred_df.csv` | Human-labeled model predictions |
| `model_outputs/experimental/seq_gene_retest_pred_df.csv` | Seq+Gene model retest predictions |
| `model_outputs/experimental/seq_gene_retested_sample_pred_df.csv` | Retested samples predictions |
| `model_outputs/Image_Model_retest_pred_df.csv` | Image model retest predictions |

**Curve Image Directory:**
| Directory | Description |
|-----------|-------------|
| `curve_imgs/` or `curve_imgs_no_axis/` | PNG images of PCR curves (referenced in training scripts) |

#### From `/media/ssd1/huong/PCR-huong/data/new_data/`

| File | Description |
|------|-------------|
| `new_groundtruth_curve_dict.pkl` | New ground truth curves |
| `new_groundtruth_target_data.csv` | New ground truth targets |
| `new_retest_curve_dict.pkl` | New retest curves |
| `new_retest_df_target_data.csv` | New retest targets |
| `new_retested_curve_dict.pkl` | New retested sample curves |
| `new_retested_df_target_data.csv` | New retested sample targets |

---

## Priority Download List

### **Critical for Evaluation (High Priority)**

1. **Model Checkpoints:**
   - `/media/ssd1/huong/PCR-huong/data/model_checkpoints/fusion_model/10_27_fusion_model_vit_delta64.pth`
   - `/media/ssd1/huong/PCR-huong/data/new_data/model_checkpoints/Seq_Model_large.ckpt`
   - `/media/ssd1/huong/PCR-huong/data/model_checkpoints/experimental/seq_gene_best.ckpt`
   - `/media/ssd1/huong/PCR-huong/data/model_checkpoints/experimental/epoch=37-step=2090.ckpt`

2. **Core Data:**
   - `/media/ssd1/huong/PCR-huong/data/data.h5` (contains curve_data, sample_info, igi_gene_call)
   - `/media/ssd1/huong/PCR-huong/data/retest_df.csv`

3. **Model Predictions (for paper plots):**
   - `/media/ssd1/huong/PCR-huong/data/model_outputs/fusion_vit_delta64_*_pred_df.csv` (all variants)
   - `/media/ssd1/huong/PCR-huong/data/model_outputs/experimental/seq_gene_*_pred_df.csv`

### **Important for Training (Medium Priority)**

4. **Curve Images:**
   - `/media/ssd1/huong/PCR-huong/data/curve_imgs_no_axis/` (entire directory)

5. **New Data Files:**
   - `/media/ssd1/huong/PCR-huong/data/new_data/new_groundtruth_curve_dict.pkl`
   - `/media/ssd1/huong/PCR-huong/data/new_data/new_groundtruth_target_data.csv`

### **Nice to Have (Low Priority)**

6. **Additional Checkpoints:**
   - Other experimental checkpoints in `model_checkpoints/experimental/`
   - Older fusion model versions

7. **Additional Predictions:**
   - Human-labeled model predictions
   - Older model output CSVs

---

## Data Schema Reference

### **curve_data** (from data.h5)
- `well_position` (str): Well position (e.g., A1, B13)
- `target` (str): Target gene (N gene, S gene, ORF1ab, MS2, E gene, RnaseP)
- `dye` (str): Dye name (FAM, VIC, JUN, ABY, ATTO 647)
- `amp_score` (float): Amplification score
- `cq` (float): Cycle threshold
- `threshold` (float): dRn threshold
- `baseline_start`, `baseline_end` (int): Baseline calculation range
- `cycle_no` (int): Cycle number (max 40 for Thermo, 45 for Luner)
- `rn` (float): Normalized reporter fluorescence
- `drn` (float): Delta Rn
- `Fn` (float): Fluorescence signal
- `pcr_plate` (str): PCR plate ID
- `curve_idx` (int): Unique curve identifier

### **sample_info** (from data.h5)
- `sample_id`, `sample_barcode` (str): Sample identifiers
- `pcr_plate` (str): PCR plate ID
- `well_position` (str): Well position
- `sample_type` (str): Sample type (clinical, control, etc.)
- `final_patient_result` (str): Final result (positive, negative, resample)
- `current_patient_result` (str): Current result (positive, negative, invalid, inconclusive)
- `created_date` (datetime): Run date
- `record_type` (str): Record type
- `retest_sample_id_1`, `retest_sample_id_2` (str): Retest sample IDs

### **igi_gene_call** (from data.h5)
- `sample_id` (str): Sample identifier
- `pcr_plate` (str): PCR plate ID
- `target` (str): Target gene
- `igi_call` (str): IGI call (positive, negative)
- `thres_ct` (float): Threshold Ct value

---

## Notes

1. **Data Versions**: The project uses multiple versions of data:
   - Original: `groundtruth_df*`
   - New: `new_groundtruth_df*`
   - No Invalid: `*_no_invalid*` (excludes invalid/inconclusive samples)

2. **Curve Data Types**:
   - `Fn`: Fluorescence signal
   - `drn`: Delta Rn (normalized)

3. **Splits**: Data is split into train/val/test in `*_split_v2*` files

4. **Model Naming Convention**:
   - `fusion`: Image + Sequence fusion
   - `vit`: Uses Vision Transformer
   - `delta64`: Includes delta sequence features (window size 64)
   - `gene`: Includes gene indicator
   - `large`: Trained on larger dataset
   - `no_invalid`: Trained excluding invalid samples

5. **Checkpoint Format**:
   - `.ckpt` files: PyTorch Lightning checkpoints (loaded with `load_from_checkpoint()`)
   - `.pth` files: Standard PyTorch checkpoints (loaded with `torch.load()`)
   - Note: One `.ckpt` file (`07_01_human_label_vit_fusioin_epoch=49-step=3050.ckpt`) is loaded into a non-Lightning model using `torch.load()['state_dict']`

5. **Evaluation Datasets**:
   - **chip60**: External ChipPCR dataset
   - **karlen**: Karlen et al. published qPCR data
   - **known**: Known dilution series
   - **retest**: Samples that were retested
   - **clinical**: Clinical samples only

---

## Download Commands

To download from remote server:

```bash
# Model checkpoints
scp -r user@server:/media/ssd1/huong/PCR-huong/data/model_checkpoints/ ./remote_checkpoints/

# Core data
scp user@server:/media/ssd1/huong/PCR-huong/data/data.h5 ./data/
scp user@server:/media/ssd1/huong/PCR-huong/data/retest_df.csv ./data/

# Model outputs
scp -r user@server:/media/ssd1/huong/PCR-huong/data/model_outputs/ ./data/

# Curve images
scp -r user@server:/media/ssd1/huong/PCR-huong/data/curve_imgs_no_axis/ ./data/

# New data
scp -r user@server:/media/ssd1/huong/PCR-huong/data/new_data/ ./data/
```

Replace `user@server` with actual credentials.
