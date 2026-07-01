# Upstream provenance

This directory vendors the official NCI Imaging Data Commons agent skill from:

- Repository: `https://github.com/ImagingDataCommons/idc-claude-skill`
- Upstream commit: `608cf464da4f798b826a3f498f0e8740fa164dc8`
- Skill version: `1.6.5`
- Vendored on: `2026-07-10`

`SKILL.md`, `LICENSE`, `references/`, and `scripts/check_version.py` are copied
verbatim from that commit. VoxelInsight does not execute the upstream installer script at
runtime; dependency versions are managed in the application's requirements file.
