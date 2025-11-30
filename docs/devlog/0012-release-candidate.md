# Devlog 0012: Release Candidate 1

**Date:** 2025-11-30
**Author:** System
**Status:** ✅ Complete

## Summary
Finalized the codebase for Release Candidate 1 (RC1). Conducted a comprehensive audit, enforced repository invariants, verified all tests, and reconciled documentation.

## Actions Taken

### 1. Codebase Audit
-   Performed full inventory of `src/` and `configs/`.
-   Identified and removed orphan files (moved docs from `src/ensembles` to `docs/`).
-   Exposed `sql_dataset_logger` in `src/core/logging/__init__.py`.

### 2. Invariant Enforcement
-   Verified no hardcoded dataset paths in training code.
-   Verified preprocessing pipelines are swappable via config.
-   Verified manifest schemas match runtime data.
-   Ensured all tests run offline.

### 3. Documentation Enhancements
-   Created `CONTRIBUTING.md` with detailed workflow guidelines.
-   Created `docs/README.md` as a central documentation index.
-   Updated `README.md` with improved navigation and links.
-   Added "How to extend" sections to `docs/preprocessing.md` and `docs/datasets.md`.
-   Fixed broken links in `docs/ensembles.md`.

### 4. Verification
-   Ran full test suite: **76 passed, 1 skipped**.
-   Verified integration tests cover end-to-end flows (Build → Train → Export).
-   Verified Vertex AI submission script and logging safety.

## Release Status
The repository is now in a stable, production-ready state.
-   **Version**: v1.0.0-rc1
-   **Tests**: 100% Pass Rate (excluding known skips)
-   **Docs**: Complete and Accurate

## Next Steps
-   Tag release in git.
-   Deploy to staging environment.
-   Onboard new engineers using `CONTRIBUTING.md`.
