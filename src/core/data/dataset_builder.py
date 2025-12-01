"""
Advanced dataset builder with state-of-the-art features.

Features:
- Stratified sampling for balanced datasets
- Class balancing strategies
- Data quality validation
- Dataset statistics and visualization
- Automatic train/val/test splitting
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


@dataclass(frozen=True)
class BuildSpec:
    dataset_id: str
    task_name: str
    dataset_version: str
    ontology_version: str
    task_mapping_version: str
    preprocess_pipeline_id: str
    preprocess_pipeline_version: str
    created_by: str
    status: str = "draft"


@dataclass(frozen=True)
class DatasetStatistics:
    """Comprehensive dataset statistics."""
    total_samples: int
    class_distribution: dict[str, int]
    split_distribution: dict[str, int]
    class_balance_ratio: float  # min_class / max_class
    missing_values: dict[str, int]
    duplicate_count: int
    baseline_stats: dict[str, Any] = None


def compute_dataset_statistics(df: pd.DataFrame, label_column: str = "label") -> DatasetStatistics:
    """
    Compute comprehensive dataset statistics.
    
    Args:
        df: Dataset DataFrame
        label_column: Name of label column
        
    Returns:
        DatasetStatistics object
    """
    # Class distribution
    class_dist = dict(df[label_column].value_counts())
    
    # Split distribution
    split_dist = dict(df.get("split", pd.Series(["unknown"] * len(df))).value_counts())
    
    # Class balance ratio
    class_counts = list(class_dist.values())
    balance_ratio = min(class_counts) / max(class_counts) if class_counts else 0.0
    
    # Missing values
    missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
    
    # Duplicates
    duplicate_count = int(df.duplicated().sum())

    # Baseline Statistics removed as per request
    baseline_stats = {}
    
    return DatasetStatistics(
        total_samples=len(df),
        class_distribution=class_dist,
        split_distribution=split_dist,
        class_balance_ratio=balance_ratio,
        missing_values=missing,
        duplicate_count=duplicate_count,
        baseline_stats=baseline_stats
    )


def stratified_split(
    df: pd.DataFrame,
    label_column: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Perform stratified train/val/test split maintaining class distribution.
    
    Args:
        df: Input DataFrame
        label_column: Column containing labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df[label_column],
        random_state=random_state,
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        stratify=temp_df[label_column],
        random_state=random_state,
    )
    
    # Add split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def balance_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    strategy: str = "oversample",
    target_samples_per_class: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance dataset using various strategies.
    
    Args:
        df: Input DataFrame
        label_column: Column containing labels
        strategy: Balancing strategy
            - "oversample": Oversample minority classes
            - "undersample": Undersample majority classes
            - "smote": Synthetic Minority Over-sampling (requires features)
        target_samples_per_class: Target number of samples per class (None = use max/min)
        random_state: Random seed
        
    Returns:
        Balanced DataFrame
    """
    class_counts = df[label_column].value_counts()
    
    if strategy == "oversample":
        # Oversample to match largest class
        target = target_samples_per_class or class_counts.max()
        
        balanced_dfs = []
        for label, count in class_counts.items():
            class_df = df[df[label_column] == label]
            if count < target:
                # Oversample
                additional = target - count
                sampled = class_df.sample(n=additional, replace=True, random_state=random_state)
                balanced_dfs.append(pd.concat([class_df, sampled]))
            else:
                balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    elif strategy == "undersample":
        # Undersample to match smallest class
        target = target_samples_per_class or class_counts.min()
        
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df[label_column] == label]
            sampled = class_df.sample(n=min(target, len(class_df)), random_state=random_state)
            balanced_dfs.append(sampled)
        
        return pd.concat(balanced_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")


def validate_dataset_quality(
    df: pd.DataFrame,
    required_columns: list[str],
    label_column: str = "label",
    min_samples_per_class: int = 10,
) -> dict[str, Any]:
    """
    Validate dataset quality and return issues.
    
    Args:
        df: Dataset DataFrame
        required_columns: List of required column names
        label_column: Label column name
        min_samples_per_class: Minimum samples required per class
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataset
    if len(df) == 0:
        issues.append("Dataset is empty")
        return {"valid": False, "issues": issues, "warnings": warnings}
    
    # Check for missing values in critical columns
    for col in required_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                warnings.append(f"Column '{col}' has {missing_count} missing values")
    
    # Check class distribution
    if label_column in df.columns:
        class_counts = df[label_column].value_counts()
        
        # Check minimum samples per class
        low_sample_classes = class_counts[class_counts < min_samples_per_class]
        if len(low_sample_classes) > 0:
            warnings.append(
                f"Classes with < {min_samples_per_class} samples: {dict(low_sample_classes)}"
            )
        
        # Check class balance
        balance_ratio = class_counts.min() / class_counts.max()
        if balance_ratio < 0.1:
            warnings.append(
                f"Severe class imbalance detected (ratio: {balance_ratio:.3f}). "
                "Consider using balancing strategies or weighted loss."
            )
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate rows")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def build_parquet_dataset(
    out_dir: str | Path,
    spec: BuildSpec,
    splits: dict[str, pd.DataFrame],
    shard_rows: int = 10_000,
    validate: bool = True,
    compute_stats: bool = True,
) -> None:
    """
    Build a task dataset with advanced features.
    
    Features:
    - Automatic validation
    - Comprehensive statistics
    - Stratified splitting support
    - Class balance analysis
    
    Args:
        out_dir: Output directory
        spec: Build specification
        splits: Dictionary of split DataFrames
        shard_rows: Rows per shard
        validate: Run quality validation
        compute_stats: Compute detailed statistics
    """
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    parquet_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    stats: dict[str, dict[str, Any]] = {}
    
    # Validate datasets if requested
    if validate:
        validation_results = {}
        for split, df in splits.items():
            result = validate_dataset_quality(
                df,
                required_columns=["image_uri"],
                label_column="label",
            )
            validation_results[split] = result
            
            if not result["valid"]:
                raise ValueError(f"Validation failed for {split} split: {result['issues']}")
            
            if result["warnings"]:
                print(f"Warnings for {split} split:")
                for warning in result["warnings"]:
                    print(f"  - {warning}")
    
    # Compute statistics if requested
    if compute_stats:
        for split, df in splits.items():
            if "label" in df.columns:
                split_stats = compute_dataset_statistics(df, label_column="label")
                stats[split] = {
                    "rows": split_stats.total_samples,
                    "class_distribution": split_stats.class_distribution,
                    "class_balance_ratio": split_stats.class_balance_ratio,
                    "missing_values": split_stats.missing_values,
                    "duplicate_count": split_stats.duplicate_count,
                    "baseline_stats": split_stats.baseline_stats,
                }
            else:
                stats[split] = {"rows": len(df)}

    # Write parquet shards
    for split, df in splits.items():
        if split not in parquet_paths:
            raise ValueError(f"Unsupported split: {split}")
        
        df = df.reset_index(drop=True)
        
        if len(df) == 0:
            continue
        
        # Shard the data
        shard_count = (len(df) + shard_rows - 1) // shard_rows
        for i in range(shard_count):
            shard = df.iloc[i * shard_rows : (i + 1) * shard_rows]
            rel = f"data/{split}-{i:04d}.parquet"
            shard.to_parquet(out_dir / rel, index=False)
            parquet_paths[split].append(rel)

    # Write manifest
    manifest = {
        "dataset_id": spec.dataset_id,
        "task_name": spec.task_name,
        "dataset_version": spec.dataset_version,
        "status": spec.status,
        "ontology_version": spec.ontology_version,
        "task_mapping_version": spec.task_mapping_version,
        "labelbox_project_mapping_versions": [],
        "sampling": {},
        "preprocess_pipeline_id": spec.preprocess_pipeline_id,
        "preprocess_pipeline_version": spec.preprocess_pipeline_version,
        "parquet_paths": parquet_paths,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": spec.created_by,
        "notes": "Generated by build_parquet_dataset with validation and statistics.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, cls=NumpyJSONEncoder), encoding="utf-8")
    
    # Write statistics
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, cls=NumpyJSONEncoder), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(
        "Use task-specific dataset builders. This module provides the file-format writer only."
    )
