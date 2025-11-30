## PRD — `ear-vision-ml` (TensorFlow + Vertex AI Experiments + Core ML, ROI-centric otoscopy)

This PRD is written to be handed to an AI implementation agent. It is **test-driven**, **documentation-driven**, and structured to minimise long-term maintenance burden while still supporting: classification, segmentation, cropper-based ROI, per-frame video inference + temporal smoothing, ensembles (cloud + iOS), local + Vertex runs, Labelbox JSON ingest, Parquet datasets in GCS, SQL dataset version logging (locked schema), and richer metadata logging in BigQuery.

### Baseline platform decision (non-negotiable)

Use the **latest TensorFlow version supported by Vertex AI custom training prebuilt containers** unless there is a proven blocker. As of the official Vertex “supported frameworks list”, TensorFlow **2.17 (Python 3.10)** is supported for **CPU and GPU** prebuilt training containers, including the `europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest` image.

---

# 1) Objectives

## 1.1 Product objectives

1. A single repo (`ear-vision-ml`) that multiple engineers can work in concurrently without tripping over each other.
2. Reproducible experiments across local and Vertex:

   * dataset identity and schema are stable;
   * preprocessing pipeline identity is versioned and logged;
   * artefacts are saved with manifests;
   * runs are tracked in Vertex Experiments and locally.
3. ROI is first-class:

   * preprocessing pipelines are swappable without touching training code;
   * cropper model is treated as a proper model with versioned artefacts and evaluation.
4. Video support is explicit and clean:

   * per-frame inference + temporal smoothing now;
   * architecture leaves room for true temporal models later without refactoring everything.

## 1.2 Non-goals (explicit)

* No network access to Labelbox APIs inside repo code or tests (only offline JSON ingestion).
* No automatic uploads of data/artefacts to your buckets (the repo can support it, but scripts must not do it by default).
* No full-blown orchestration platform (TFX/Kubeflow). Keep it lean.

---

# 2) Target users & workflows (what the repo must support)

## 2.1 Personas

* ML engineers (training, evaluation, exports, experiments)
* Data/ML platform engineer (dataset ingestion/building, registries)
* iOS engineer (Core ML artefacts + strict output contracts)

## 2.2 Core workflows

1. **Ingest Labelbox exports** (JSON on disk) → canonical label tables (Parquet) → task datasets (Parquet) in GCS layout.
2. **Train ROI cropper** locally or on Vertex, export to Core ML for device ROI coordinates.
3. **Train downstream models** (classification/segmentation) on ROI-cropped inputs.
4. **Run experiments** locally and on Vertex using **the same entrypoints** and **Hydra configs**.
5. **Track runs** in Vertex Experiments (plus local run records).
6. **Export models** (SavedModel + TFLite; Core ML when enabled).
7. **Video inference** from URI+timestamps; temporal smoothing and reports.

---

# 3) Architectural principles (to avoid an unmaintainable monster)

## 3.1 Minimal moving parts

* Parquet stays the canonical dataset artefact format (with manifest + stats).
* SQL stays, schema unchanged, used only for “dataset version logging requirement”.
* BigQuery becomes the flexible registry/log store (datasets, runs, models) because SQL is locked.
* Hydra configs are the public API for experiments and reproducibility.

## 3.2 Hard separation of concerns

* **Models** do not contain video logic.
* **Preprocessing** does not live scattered across notebooks or ad-hoc code; it is versioned and tested.
* **Datasets** are immutable once published; changes create new dataset versions, not silent overwrites.

---

# 4) Repo structure (must be created exactly)

```text
ear-vision-ml/
  config/
    env/
      conda-tf217.yml
    docker/
      tf217/
        Dockerfile
        build.sh
        push.sh

  configs/
    config.yaml
    task/
      cropper.yaml
      classification.yaml
      segmentation.yaml
      video_runtime.yaml
    preprocess/
      full_frame_v1.yaml
      cropper_model_v1.yaml
      cropper_fallback_v1.yaml
    model/
      cropper_mobilenetv3.yaml
      cls_mobilenetv3.yaml
      seg_unet.yaml
    data/
      local.yaml
      gcs_parquet.yaml
      labelbox_json.yaml
      sampling.yaml
      augmentations.yaml
    training/
      default.yaml
      mixed_precision.yaml
      distributed.yaml
      hypertune.yaml
    export/
      tflite.yaml
      coreml.yaml
    evaluation/
      default.yaml
      video.yaml
    ensemble/
      cloud_soft_voting.yaml
      device_swift_v1.yaml

  src/
    core/
      contracts/
        ontology.yaml
        labelbox_mappings/
        task_mappings/
        dataset_manifest_schema.json
        model_manifest_schema.json
        roi_contract.py
      data/
        labelbox_ingest.py
        dataset_builder.py
        dataset_loader.py
        media_reader.py
        parquet_schema.py
      preprocess/
        registry.py
        pipelines/
          full_frame_v1.py
          cropper_model_v1.py
          cropper_fallback_v1.py
        debug_viz.py
      models/
        backbones/
        heads/
        factories/
          model_factory.py
      training/
        trainer_base.py
        losses.py
        metrics.py
        callbacks.py
      logging/
        local_logger.py
        vertex_experiments.py
        bq_logger.py
        sql_dataset_logger.py
      export/
        exporter.py
        equivalence.py
      utils/

    tasks/
      cropper/
        entrypoint.py
        trainer.py
        evaluation.py
      classification/
        entrypoint.py
        trainer.py
        evaluation.py
      segmentation/
        entrypoint.py
        trainer.py
        evaluation.py

    runtimes/
      video_inference/
        frame_sampler.py
        temporal_aggregators.py
        offline_runner.py

    ensembles/
      cloud_runtime.py
      device_contract.md
      distillation.md

  scripts/
    local_train.sh
    vertex_submit.sh
    build_dataset.sh
    export_model.sh
    run_video_inference.sh

  tests/
    fixtures/
      images/
      labelbox_exports/
      manifests/
    unit/
      test_roi_contract.py
      test_preprocess_registry.py
      test_dataset_manifest_schema.py
      test_model_factory.py
    integration/
      test_dataset_build_smoke.py
      test_training_smoke.py
      test_export_smoke.py
      test_video_runtime_smoke.py

  docs/
    README.md
    repo_rules.md
    datasets.md
    preprocessing.md
    experiments.md
    deployment_ios.md
    ensembles.md
    devlog/
    adr/

  README.md
  pyproject.toml (or requirements.in + requirements.txt, see §5.2)
  ruff.toml
  mypy.ini
```

---

# 5) Technologies, tooling, and versions (explicit)

## 5.1 Training/inference frameworks

* TensorFlow/Keras: **2.17.x** (baseline for both local and Vertex)
* tf.data for pipelines; Keras model subclassing only where justified.

## 5.2 Dependency management (reduce Conda mess)

**Requirement:** one reproducible lock for Python dependencies used for local + Docker.

Choose one and implement it cleanly:

* Option A (recommended for simplicity): `requirements.in` + `pip-tools` to generate `requirements.txt` lock.
* Option B: `pyproject.toml` (Poetry) if your team is already comfortable.

Conda env (`conda-tf217.yml`) must install from the lock (pip section). No separate floating installs.

## 5.3 Configuration

* Hydra: `hydra-core`
* OmegaConf
* Configs live in `configs/` as the “experiment API”.

## 5.4 Cloud integrations (no network in tests)

* Vertex Experiments via `google-cloud-aiplatform`
* BigQuery via `google-cloud-bigquery`
* GCS via `google-cloud-storage` (used in runtime code, but never in tests).

## 5.5 Data formats

* Task datasets: Parquet
* Manifests/stats: JSON
* Artefacts: SavedModel, TFLite; Core ML optional/flagged

## 5.6 Video/frame handling

* Default: OpenCV (`opencv-python-headless`) OR `imageio` (agent must pick one and standardise).
* No network decode or remote video streaming in tests; fixtures are local.

## 5.7 Quality gates

* `pytest`
* `ruff` (lint)
* `mypy` (type checks for core modules)
* basic CPU-only smoke trainings in CI

---

# 6) Execution policy for implementation agent

## 6.1 Mandatory “document-as-you-build”

The agent must **document every meaningful change** as it implements the repo.

### Requirements

1. Create `docs/devlog/` and maintain a running set of markdown files:

   * `docs/devlog/0001-repo-bootstrap.md`
   * `docs/devlog/0002-config-system.md`
   * `docs/devlog/0003-dataset-manifest-and-loader.md`
   * etc.
2. Each devlog document must include:

   * What was implemented
   * Files created/modified
   * How to run it (commands)
   * Tests added/updated (and how to run them)
   * Known limitations and next steps
3. Create `docs/adr/` (Architecture Decision Records) and write ADRs for:

   * repo structure and ownership boundaries
   * dataset manifest schema
   * ROI preprocessing contract
   * TF/container baseline on Vertex
4. No big-bang implementations. Each PR-sized step must be documented and tested.

## 6.2 Prohibited actions

* No network calls in tests.
* No uploads to GCS/Labelbox/Vertex as part of test runs.
* No changes to developer machine outside conda env creation.
* No silent schema changes.

---

# 7) Dataset system requirements (aligned to your current reality)

## 7.1 Current reality honoured

* Data rows reference media via **URI + timestamps** (sampling on the fly).
* SQL logs only dataset versions (locked).

## 7.2 Dataset artefact layout (GCS and local mirroring)

**Requirement:** each dataset is a folder with:

* `data/` Parquet shards by split
* `manifest.json` (single source of truth)
* `stats.json`

Example layout:

```text
gs://<bucket>/datasets/<task>/<dataset_id>/
  data/
    train-0000.parquet
    val-0000.parquet
    test-0000.parquet
  manifest.json
  stats.json
```

## 7.3 Dataset manifest (must follow schema)

The manifest must explicitly capture:

* `dataset_id`, `task_name`, `version`, `status`
* label mapping versions (ontology, labelbox-mapping(s), task-mapping)
* sampling configuration hash + parameters
* preprocessing pipeline id/version (see §8)
* parquet split shard list
* created_by, created_at

Schema enforced by `dataset_manifest_schema.json`; tests must validate it.

## 7.4 Registry logging

* SQL: log `dataset_version` (existing behaviour; minimal adapter in `sql_dataset_logger.py`)
* BigQuery: full manifest + stats + build lineage (rich logging)

---

# 8) ROI preprocessing requirements (your critical differentiator)

## 8.1 ROI contract (hard)

Define in `roi_contract.py`:

* `bbox_xyxy_norm`: `[x1, y1, x2, y2]` in [0,1]
* `confidence`: [0,1]
* `source`: `cropper|fallback|full_frame`
* Validation + clipping rules

Unit tests must cover:

* invalid boxes
* clipping behaviour
* area sanity checks

## 8.2 Preprocess pipeline interface (swappable)

All pipelines implement:

* `apply(image, metadata, cfg) -> (image_out, metadata_out)`
* output includes ROI fields even if full-frame (“full_frame” source with bbox=full image)

## 8.3 Pipelines in MVP

* `full_frame_v1`: resize/normalise only
* `cropper_model_v1`: runs cropper (TF runtime) to produce ROI bbox
* `cropper_fallback_v1`: uses cropper when confident, else deterministic fallback (centre crop or rule-based)

## 8.4 Debugging artefacts

Preprocess module must be able to produce:

* overlay images showing bbox on original + cropped result (saved under run artefacts)
* bbox distribution stats logged per run

---

# 9) Modelling requirements

## 9.1 Tasks in scope (MVP)

1. **Cropper task**

   * model outputs bbox + confidence (or equivalent)
   * exports to Core ML if enabled
2. **Classification task**

   * baseline MobileNetV3 backbone
3. **Segmentation task**

   * baseline U-Net

## 9.2 Model factory

A single `model_factory.py` that builds models from config. Unit tests must build each registered model and run a forward pass on dummy tensors.

## 9.3 Loss/metrics minimum set

* Classification: accuracy, macro F1, per-class precision/recall
* Segmentation: Dice, IoU, pixel accuracy
* Cropper: bbox regression loss + sanity metrics (invalid rate, mean area)

---

# 10) Video runtime requirements (current + future-proof)

* Input: list of (URI, timestamp range) or a dataset_id containing URI+timestamps
* `frame_sampler.py`: deterministic sampling strategy per config
* `temporal_aggregators.py`: moving average + hysteresis + majority vote
* Output: JSON report saved under artefacts; integration test uses local fixtures.

---

# 11) Ensembles requirements (cloud + device)

* Cloud ensemble runtime combines multiple model outputs per config (soft voting / weighted).
* Device:

  * Support the current approach: cropper Core ML → Swift crop → downstream model(s)
  * Document a strict Swift interface contract in `device_contract.md`:

    * bbox schema, tensor ordering, class ordering, threshold conventions
  * Distillation remains a documented option (`distillation.md`) not required in MVP.

---

# 12) Experiment tracking requirements (Vertex + local)

## 12.1 Vertex Experiments

* Log params, metrics, and artefact URIs into Experiment Runs using Vertex SDK.
* Must work in Vertex jobs; locally it should either:

  * log if authenticated, or
  * degrade gracefully and still log locally.

## 12.2 Local run records (mandatory)

Every run produces:

* `artifacts/runs/<run_id>/run.json`
* includes resolved Hydra config, git revision, dataset_id, preprocess id/version, model id/version, output artefacts, key metrics.

---

# 13) Test-driven delivery plan (the agent must follow this order)

## Phase 0 — Repo bootstrap (tests + docs first)

* Create structure, env files, Dockerfile.
* Add ruff/mypy/pytest configuration.
* Write `docs/devlog/0001-repo-bootstrap.md` and ADR 0001 (repo structure).
* Implement and pass minimal tests:

  * `test_roi_contract` (even if pipelines not complete yet)
  * `test_dataset_manifest_schema` (schema validator)

## Phase 1 — Config system + entrypoints

* Hydra config tree working.
* Task entrypoints print resolved config.
* Document in devlog and ADR for config strategy.

## Phase 2 — Dataset manifest + loader (local fixtures)

* Implement manifest schema + loader reading local fixture manifests/parquet.
* Smoke test: load dataset and iterate a few records.

## Phase 3 — Preprocess pipelines (full_frame + cropper_fallback)

* Implement preprocess registry and pipelines.
* Unit tests for pipeline swapping.
* Debug overlay artefact generation.

## Phase 4 — Model factory + trainers (smoke training)

* Implement baseline models.
* Training smoke test: 2 steps on fixture dataset.
* Local artefacts created and validated.

## Phase 5 — Export (SavedModel + TFLite + equivalence)

* Export smoke test that converts and runs TFLite inference.

## Phase 6 — Vertex Experiments logging + scripts

* Implement logging module; add a “no-auth local mode”.
* Provide scripts for Vertex submission using tf-gpu.2-17.py310 container.

## Phase 7 — Video inference runtime

* Frame sampler + temporal aggregator.
* Integration smoke test using local fixtures.

**At each phase**:

* update a devlog markdown file
* update relevant docs pages
* ensure tests pass

---

# 14) Documentation deliverables (must exist on completion)

* `docs/repo_rules.md`: contribution rules, ownership boundaries, how to add models/pipelines/tasks
* `docs/datasets.md`: schemas, manifest fields, build and promotion rules
* `docs/preprocessing.md`: ROI contract, pipeline catalogue, versioning, debug outputs
* `docs/experiments.md`: local vs Vertex runs; Vertex Experiments integration
* `docs/deployment_ios.md`: device pipeline contract (cropper → Swift crop → downstream)
* `docs/ensembles.md`: cloud vs device ensemble behaviour + constraints
* `docs/devlog/*`: implementation progress log
* `docs/adr/*`: architecture decisions

---

# 15) Acceptance criteria (binary)

1. `pytest` passes locally with **no network access**.
2. Local runs:

   * classification smoke train runs and produces artefacts + local run record
   * segmentation smoke train runs and produces artefacts + local run record
3. Preprocessing pipeline is swappable via Hydra config without code changes.
4. Export produces SavedModel + TFLite(s) and a `model_manifest.json`.
5. Vertex submission script references official TF 2.17 prebuilt training containers and training code supports Vertex Experiments logging.
6. Documentation exists and is coherent; devlog and ADRs reflect actual implementation steps.
