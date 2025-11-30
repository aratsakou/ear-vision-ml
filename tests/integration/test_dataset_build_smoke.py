from pathlib import Path

import pytest

from src.core.data.dataset_loader import load_dataset_from_manifest_dir


@pytest.fixture
def fixture_manifest_dir() -> Path:
    return Path("tests/fixtures/manifests/local_smoke")

def test_load_dataset_smoke(fixture_manifest_dir: Path) -> None:
    """Smoke test for loading a dataset from a local fixture."""
    ds = load_dataset_from_manifest_dir(fixture_manifest_dir, split="train", batch_size=2)
    
    # Take one batch
    batch = next(iter(ds))
    
    assert isinstance(batch, dict)
    assert "image_uri" in batch
    assert "label" in batch
    assert batch["image_uri"].shape[0] == 2
    
def test_load_dataset_invalid_split(fixture_manifest_dir: Path) -> None:
    """Test loading an invalid split."""
    with pytest.raises(ValueError, match="Split 'invalid' not found"):
        load_dataset_from_manifest_dir(fixture_manifest_dir, split="invalid")
