# MU-Glioma-Post nnU-Net ResEnc-L five-fold cross-validation

This package applies the completed nnU-Net ResEnc-L protocol to patient-grouped outer five-fold cross-validation using exactly the same outer folds as the completed 3D U-Net, SegResNet, and MedNeXt-S CV runs.

## Frozen methodology

- Cohort: finalized V5 MU-Glioma-Post registration cache and labels.
- Eligibility: 353 review-cleared scan pairs from 146 patients; the 38 automatic registration-QC review-required pairs are excluded without changing any label.
- Outer CV: one immutable, balanced patient registry with five held-out folds. Every eligible pair is evaluated exactly once.
- Per fold: approximately 70% training, 10% inner validation, and 20% held-out outer testing by patient.
- Inputs: earlier and registered recurrent T1, T1CE, T2, and FLAIR.
- Networks: two independent native nnU-Net v2.8.1 Residual Encoder U-Net Large specialists. Each is reconstructed from the checkpoint's embedded `nnUNetResEncUNetLPlans` and strictly initialized from the audited BraTS `checkpoint_best.pth`.
- Targets: earlier whole tumor and recurrent whole tumor.
- Loss: equal timepoint weighting; each specialist uses 0.70 Dice loss + 0.30 BCEWithLogits loss.
- Infiltration prediction: deterministic finalized V5 2-20 mm rule applied to the two predicted whole-tumor masks.
- Training: 30 maximum epochs, 480 sampled pairs per epoch, BF16 when supported, AdamW, cosine schedule, minimum 8 epochs, patience 6. Each fold automatically probes physical pair batch 8/4/2/1 and uses gradient accumulation when necessary to preserve effective pair batch 8.
- Checkpoint selection: best of pretrained epoch 0 and finetuned epochs using inner validation only.
- Reporting: per-fold test metrics, pooled out-of-fold metrics, fold mean and standard deviation, patient-bootstrap 95% CIs, volume-stratified metrics, lesion/surface metrics, one-sided oracle analysis, predictions, and QC for all 353 held-out predictions.

## Two-GPU execution

Two folds run concurrently, one fold per GPU. This intentionally avoids DataParallel/DDP. When either fold finishes, that physical GPU automatically receives the next pending fold.

## Install and run

```bash
V5=$GLIOMA_PROJECT_ROOT
V4=$GLIOMA_PRETRAINING_ROOT
P="$V5/finetuning/scripts/nnunet_mu_5fold_cv_v1"
NNPY="$V4/environments/nnunetv2_2_8_1/bin/python"

mkdir -p "$V5/finetuning/scripts"
tar -xzf /path/to/workspace/44_nnunet_mu_5fold_cv_v1.tar.gz \
  -C "$V5/finetuning/scripts"

cat "$P/VERSION.txt"
"$NNPY" -u "$P/self_test_v1.py"
bash "$P/prepare_fresh_start.sh"
CUDA_VISIBLE_DEVICES=0,1 bash "$P/launch_v1.sh"
```

The launch is detached. Rerunning the same launch command resumes incomplete folds from their latest checkpoints after a disconnection or GPU-allocation loss.

## Monitor

```bash
watch -n 20 "$NNPY" -u "$P/monitor_v1.py" --once
```

Detailed live progress for one fold, with each completed epoch retained on its own terminal line:

```bash
"$NNPY" -u "$P/live_epoch_v1.py" --fold 0
```

Logs:

```bash
EXP="$V5/finetuning/experiments/nnUNetResEncL_MU_5fold_CV_v1"
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
$GLIOMA_PROJECT_ROOT/finetuning/experiments/nnUNetResEncL_MU_5fold_CV_v1/aggregate
```
