from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Trial:
    id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: str

class HyperparameterTuner(ABC):
    """Interface for hyperparameter tuning services (e.g., Vertex AI Vizier)."""
    
    @abstractmethod
    def create_study(self, study_config: Dict[str, Any]) -> str:
        """Create a new study and return its ID."""
        pass
    
    @abstractmethod
    def get_suggestions(self, study_id: str, count: int = 1) -> List[Trial]:
        """Get suggested trials."""
        pass
    
    @abstractmethod
    def report_metrics(self, trial_id: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Report intermediate metrics for a trial."""
        pass
    
    @abstractmethod
    def complete_trial(self, trial_id: str, final_measurement: Dict[str, float], infeasible: bool = False) -> None:
        """Mark a trial as complete with final measurement."""
        pass
