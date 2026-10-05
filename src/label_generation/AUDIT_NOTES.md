# Public-export audit notes

- The active workflow is exact MU v2 only.
- Runtime paths are configured with environment variables in `path_config.py`.
- Writes are limited to the configured project, label, control, and temporary roots.
- Raw images, derived NIfTI files, private manifests, identifiers, and case-level QC are excluded.
- The example manifest contains synthetic values and documents schema only.
