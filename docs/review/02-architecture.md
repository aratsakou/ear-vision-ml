# Architecture Overview

## What does this repository do?

`ear-vision-ml` is a machine learning framework for analyzing medical ear images (otoscopy). It handles the entire lifecycle of a machine learning model:
1.  **Data**: Ingesting raw images and creating structured datasets (Parquet).
2.  **Training**: Teaching models to classify diseases or segment ear parts.
3.  **Export**: Converting trained models into formats ready for mobile apps (TFLite, CoreML).
4.  **Evaluation**: Checking how well the models perform.

## The "Mental Model"

Think of this repo as a **factory line**.
- Raw materials (images) come in.
- They get processed (resized, augmented).
- They go into a machine (the Trainer).
- The machine produces a product (the Model).
- The product is packaged (Exported) for shipping.

Everything is controlled by **Configuration** (Hydra). You don't change the code to change the batch size; you change the config.

## Key Components

```ascii
[Config (Hydra)] --> [Dependency Injection Container]
                            |
                            v
[Dataset Builder] --> [Data Loader] --> [Model Builder] --> [Trainer] --> [Exporter]
       ^                  ^                  ^                 ^              ^
       |                  |                  |                 |              |
   Raw Images        Parquet Files      Architecture      Loss/Metrics    TFLite/CoreML
```

### 1. Configuration (Hydra)
We use **Hydra** to manage settings. Instead of passing 50 arguments to a script, we use YAML files.
- `configs/config.yaml`: The master switchboard.
- `configs/task/`: Defines what we are doing (e.g., `classification`).
- `configs/model/`: Defines the brain (e.g., `efficientnet`).

### 2. Dependency Injection (DI)
We use a **Container** (`src/core/di.py`) to wire everything together.
- Instead of `Trainer` creating a `Model`, the Container gives the `Model` to the `Trainer`.
- This makes it easy to swap parts (e.g., use a different Model) without rewriting the Trainer.

### 3. The Lifecycle

#### A. Dataset Creation
**Goal**: Turn messy folders of images into a clean, structured file.
- **Tool**: `scripts/build_otoscopic_dataset.py`
- **Output**: A folder with `.parquet` files (tables of image paths and labels) and a `manifest.json`.

#### B. Training
**Goal**: Learn from the data.
- **Input**: The `manifest.json` from step A.
- **Process**:
    1.  **Load**: Read images, resize, apply augmentations (random rotations, flips).
    2.  **Build**: Create the neural network (e.g., MobileNet).
    3.  **Train**: Loop through data, update weights.
- **Output**: A saved Keras model (`saved_model/`).

#### C. Export
**Goal**: Make the model run on a phone.
- **Input**: The trained Keras model.
- **Process**:
    1.  **Quantize**: Shrink the model (make it smaller and faster) using TFLite.
    2.  **Convert**: Create `.tflite` and `.mlpackage` (CoreML) files.
- **Output**: `model.tflite`, `model.mlpackage`.

## Why do we do things this way?

- **Manifests**: So we know exactly what data went into a model. "Garbage in, garbage out."
- **Hydra**: So we can run 100 experiments with different settings without changing code.
- **DI**: So we can test parts in isolation (unit testing).
