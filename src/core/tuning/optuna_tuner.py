from typing import Any, Dict, List, Optional
import logging
import optuna
from optuna.samplers import GPSampler
from src.core.tuning.hyperparam_tuner import HyperparameterTuner, Trial

log = logging.getLogger(__name__)

class OptunaTuner(HyperparameterTuner):
    """
    Optuna-based implementation of HyperparameterTuner.
    Uses GPSampler for Gaussian Process Bayesian Optimization.
    """
    
    def __init__(self, storage: str = "sqlite:///db.sqlite3"):
        self.storage = storage
        self.studies = {} # Cache study objects
        self.param_specs = {} # Cache parameter specs per study

    def create_study(self, study_config: Dict[str, Any]) -> str:
        study_name = study_config.get("display_name", "ear-vision-study")
        direction = study_config.get("metrics", {}).get("goal", "MAXIMIZE").lower()
        if direction == "maximize":
            direction = "maximize"
        else:
            direction = "minimize"
            
        # Use GPSampler for Gaussian Optimization
        sampler = GPSampler()
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction=direction,
            sampler=sampler,
            load_if_exists=True
        )
        
        # Store parameter specs for get_suggestions
        self.param_specs[study_name] = study_config.get("parameters", {})
        self.studies[study_name] = study
        
        log.info(f"Created/Loaded Optuna study: {study_name}")
        return study_name

    def get_suggestions(self, study_id: str, count: int = 1) -> List[Trial]:
        if study_id not in self.studies:
            # Try to load if not in cache
            try:
                self.studies[study_id] = optuna.load_study(study_name=study_id, storage=self.storage)
            except KeyError:
                raise ValueError(f"Study {study_id} not found")
        
        study = self.studies[study_id]
        specs = self.param_specs.get(study_id, {})
        
        trials = []
        for _ in range(count):
            # Ask Optuna for a new trial
            # In Optuna's ask-and-tell interface, we get a trial object
            # and then we must define the search space on it.
            optuna_trial = study.ask()
            
            params = {}
            for param_name, spec in specs.items():
                p_type = spec.get("type", "DOUBLE")
                
                if p_type == "DOUBLE":
                    min_val = spec.get("min_value", 0.0)
                    max_val = spec.get("max_value", 1.0)
                    log_scale = spec.get("scale", "linear") == "log"
                    params[param_name] = optuna_trial.suggest_float(param_name, min_val, max_val, log=log_scale)
                    
                elif p_type == "INTEGER":
                    min_val = spec.get("min_value", 0)
                    max_val = spec.get("max_value", 100)
                    log_scale = spec.get("scale", "linear") == "log"
                    params[param_name] = optuna_trial.suggest_int(param_name, min_val, max_val, log=log_scale)
                    
                elif p_type == "CATEGORICAL":
                    choices = spec.get("values", [])
                    params[param_name] = optuna_trial.suggest_categorical(param_name, choices)
                    
                elif p_type == "DISCRETE":
                    choices = spec.get("values", [])
                    # Treat discrete as categorical or float? Usually discrete numbers.
                    # Optuna suggest_float with step, or suggest_categorical.
                    # Let's use categorical for simplicity if values are provided list
                    params[param_name] = optuna_trial.suggest_categorical(param_name, choices)

            trials.append(Trial(
                id=str(optuna_trial.number),
                parameters=params,
                metrics={},
                status="ACTIVE"
            ))
            
        return trials

    def report_metrics(self, trial_id: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # Optuna doesn't easily support reporting metrics for a trial ID without the trial object
        # unless we reload the trial.
        # But since we are running locally, we might have the trial object if we kept it?
        # Or we can just ignore intermediate reporting for now if using ask-and-tell across processes.
        # However, if we are in the same process (which we are for local tuning script), we can use the study.
        pass

    def complete_trial(self, trial_id: str, final_measurement: Dict[str, float], infeasible: bool = False) -> None:
        # We need to find the study this trial belongs to.
        # For simplicity, we assume we only have one active study or we search.
        # Or we pass study_id in the context.
        # But the interface only has trial_id.
        # Let's assume the last accessed study or iterate.
        
        # In a real implementation, we'd map trial_id to study.
        # Here, we'll iterate over cached studies.
        
        target_study = None
        for study in self.studies.values():
            try:
                # Check if trial exists in this study (expensive?)
                # Optimization: The trial_id is usually an integer.
                # We can try to tell.
                study.tell(int(trial_id), list(final_measurement.values())[0]) # Assume single objective for now
                target_study = study
                break
            except Exception:
                continue
        
        if not target_study:
            log.warning(f"Could not find study for trial {trial_id} to complete.")
