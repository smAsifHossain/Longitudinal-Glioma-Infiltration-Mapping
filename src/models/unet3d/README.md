# MU-Glioma-Post 3D U-Net five-fold cross-validation

This package converts the completed fixed-split 3D U-Net pipeline into patient-grouped outer five-fold cross-validation without reusing the previous locked-test experiment as a fold.

## Frozen methodology

- Cohort: finalized V5 MU-Glioma-Post registration cache and labels.
- Eligibility: 353 review-cleared scan pairs from 146 patients; the 38 automatic registration-QC review-required pairs are excluded without changing any label.
- Outer CV: one immutable, balanced patient registry with five held-out folds. Every eligible pair is evaluated exactly once.
- Per fold: approximately 70% training, 10% inner validation, and 20% held-out outer testing by patient.
- Inputs: earlier and registered recurrent T1, T1CE, T2, and FLAIR.
- Networks: two independent MONAI 3D U-Nets initialized from the completed 60-epoch BraTS checkpoint.
- Targets: earlier whole tumor and recurrent whole tumor.
- Loss: equal timepoint weighting; each specialist uses 0.70 Dice loss + 0.30 BCEWithLogits loss.
- Infiltration prediction: deterministic finalized V5 2-20 mm rule applied to the two predicted whole-tumor masks.
- Training: 30 maximum epochs, 480 sampled pairs per epoch, BF16 when supported, AdamW, cosine schedule, minimum 8 epochs, patience 6.
- Checkpoint selection: best of pretrained epoch 0 and finetuned epochs using inner validation only.
- Reporting: per-fold test metrics, pooled out-of-fold metrics, fold mean and standard deviation, patient-bootstrap 95% CIs, volume-stratified metrics, lesion/surface metrics, one-sided oracle analysis, predictions, and QC for all 353 held-out predictions.

## Two-GPU execution

Two folds run concurrently, one fold per GPU. This intentionally avoids DataParallel/DDP. When either fold finishes, that physical GPU automatically receives the next pending fold.

## Install and run

```bash
V5=$GLIOMA_PROJECT_ROOT
P="$V5/finetuning/scripts/unet_mu_5fold_cv_v1"

mkdir -p "$V5/finetuning/scripts"
tar -xzf /path/to/workspace/40_unet_mu_5fold_cv_v1.tar.gz \
  -C "$V5/finetuning/scripts"

cat "$P/VERSION.txt"
python -u "$P/self_test_v1.py"
bash "$P/prepare_fresh_start.sh"
CUDA_VISIBLE_DEVICES=0,1 bash "$P/launch_v1.sh"
```

The launch is detached. Rerunning the same launch command resumes incomplete folds from their latest checkpoints after a disconnection or GPU-allocation loss.

## Monitor

```bash
watch -n 20 python -u "$P/monitor_v1.py" --once
```

Detailed live progress for one fold, with each completed epoch retained on its own terminal line:

```bash
python -u "$P/live_epoch_v1.py" --fold 0
```

Logs:

```bash
EXP="$V5/finetuning/experiments/UNet3D_MU_5fold_CV_v1"
tail -f "$EXP/manager.log"
tail -f "$EXP/fold_0/training.log"
tail -f "$EXP/fold_0/test.log"
```

Safe stop:

```bash
bash "$P/stop_v1.sh"
```

Final aggregate results are written under:

```text
$GLIOMA_PROJECT_ROOT/finetuning/experiments/UNet3D_MU_5fold_CV_v1/aggregate
```
