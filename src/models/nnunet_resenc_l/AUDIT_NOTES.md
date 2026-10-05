# Five-fold conversion audit

- The earlier fixed 70/15/15 experiment is not counted as a CV fold and is never overwritten.
- A single SHA-256-locked outer-fold registry is generated before training.
- Assignment is by patient, balancing patient counts, scan-pair counts, and the fixed infiltration-volume bins.
- Each fold has a distinct inner validation set; held-out outer-test images, masks, labels, and metrics are not used during training.
- Training reads only outer-test case and patient identifiers for leakage verification.
- Every eligible patient and scan pair is held out exactly once across the five folds.
- The original finalized V5 label/cache consistency audit and real-data GPU forward/backward smoke test are retained.
- The fold registry, patch sampling, augmentation, optimizer, loss, epoch limit, early stopping, threshold, inference, metrics, and QC layout match the completed 3D U-Net CV comparison.
- The only scientific model change is the native nnU-Net v2.8.1 Residual Encoder U-Net Large architecture and its audited pretrained checkpoint.
- The network is rebuilt directly from the checkpoint's embedded plans, dataset JSON, and `3d_fullres` configuration. No MONAI approximation or weight remapping is used.
- Strict initialization verifies the trainer name, `nnUNetResEncUNetLPlans`, four-channel order, WT/TC/ET regions, 160 x 192 x 160 patch, tensor compatibility, finite weights, pretrained epoch, and best EMA pseudo-Dice.
- The launcher and every child process run inside the audited nnU-Net v2.8.1 Python environment with nnU-Net compilation disabled.
- Preflight requires and verifies exact case, patient, and outer-fold equality with the completed 3D U-Net five-fold registry.
- Each GPU independently probes physical pair batch 8/4/2/1; gradient accumulation preserves effective pair batch 8 and 60 optimizer updates per epoch.
- Two GPUs execute independent folds concurrently. There is no multi-GPU model replication, NCCL, DataParallel, or DDP.
- A signal-triggered stop checkpoints the active optimization step. An abrupt allocation loss resumes from the latest periodic checkpoint.
- Final reporting includes fold mean +/- standard deviation and a pooled out-of-fold estimate with patient-level bootstrap confidence intervals.
