import pytest
from unittest.mock import MagicMock, patch
from src.core.tuning.vertex_vizier import VertexVizierTuner

def test_vertex_vizier_tuner_report_metrics_local():
    """Test that report_metrics prints to stdout when hypertune is missing (local mode)."""
    tuner = VertexVizierTuner(project="test-project", location="us-central1")
    
    with patch("builtins.print") as mock_print:
        tuner.report_metrics("trial-1", {"val_accuracy": 0.85}, step=1)
        
        mock_print.assert_called_with("[Local/No-Hypertune] Reporting metrics for trial trial-1: {'val_accuracy': 0.85}")

@patch("src.core.tuning.vertex_vizier.aiplatform")
def test_vertex_vizier_tuner_init(mock_aiplatform):
    """Test tuner initialization."""
    tuner = VertexVizierTuner(project="test-project", location="us-central1")
    mock_aiplatform.init.assert_called_with(project="test-project", location="us-central1")
