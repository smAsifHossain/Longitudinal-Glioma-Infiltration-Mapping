# Five-fold conversion audit

- The earlier fixed 70/15/15 experiment is not counted as a CV fold and is never overwritten.
- A single SHA-256-locked outer-fold registry is generated before training.
- Assignment is by patient, balancing patient counts, scan-pair counts, and the fixed infiltration-volume bins.
- Each fold has a distinct inner validation set; held-out outer-test images, masks, labels, and metrics are not used during training.
- Training reads only outer-test case and patient identifiers for leakage verification.
- Every eligible patient and scan pair is held out exactly once across the five folds.
- The original finalized V5 label/cache consistency audit and real-data GPU forward/backward smoke test are retained.
- The architecture, checkpoint initialization, patch sampling, augmentation, optimizer, loss, epoch limit, early stopping, threshold, inference, metrics, and QC layout match the completed 3D U-Net experiment.
- Two GPUs execute independent folds concurrently. There is no multi-GPU model replication, NCCL, DataParallel, or DDP.
- A signal-triggered stop checkpoints the active optimization step. An abrupt allocation loss resumes from the latest periodic checkpoint.
- Final reporting includes fold mean +/- standard deviation and a pooled out-of-fold estimate with patient-level bootstrap confidence intervals.
