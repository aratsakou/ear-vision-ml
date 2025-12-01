from typing import Any, Dict, List, Optional
import logging
import keras_tuner
from keras_tuner.oracles import BayesianOptimizationOracle
from keras_tuner.engine.trial import Trial as KerasTrial
from src.core.tuning.hyperparam_tuner import HyperparameterTuner, Trial

log = logging.getLogger(__name__)

class KerasTunerImpl(HyperparameterTuner):
    """
    Keras Tuner-based implementation of HyperparameterTuner.
    Uses BayesianOptimizationOracle for Gaussian Process-based optimization.
    """
    
    def __init__(self, project_name: str = "ear_vision_tuning", directory: str = "tuning_results"):
        self.project_name = project_name
        self.directory = directory
        self.oracles = {} # Cache oracle objects per study

    def create_study(self, study_config: Dict[str, Any]) -> str:
        study_name = study_config.get("display_name", "default_study")
        objective_name = "val_acc" # Default, should be configurable
        direction = study_config.get("metrics", {}).get("goal", "MAXIMIZE").lower()
        
        # Keras Tuner uses 'max' or 'min'
        mode = "max" if direction == "maximize" else "min"
        
        objective = keras_tuner.Objective(name=objective_name, direction=mode)
        
        # Initialize Bayesian Optimization Oracle
        # max_trials is handled by the loop, but Oracle needs a limit or we just keep asking.
        # We set a high number here and control loop externally.
        oracle = BayesianOptimizationOracle(
            objective=objective,
            max_trials=1000,
            num_initial_points=5, # Random search first
            alpha=1e-4,
            beta=2.6,
            seed=42
        )
        
        # We need to set the project name for the oracle to manage state
        oracle.project_name = study_name
        oracle._set_project_dir(self.directory, study_name)
        
        # Store parameter specs to define search space when asked
        # Keras Tuner defines space dynamically in 'populate_space', 
        # but since we are adapting to a fixed config-based space, 
        # we will use a HyperParameters object and pre-define it if possible,
        # or define it during the first call.
        self.oracles[study_name] = {
            "oracle": oracle,
            "specs": study_config.get("parameters", {})
        }
        
        log.info(f"Created Keras Tuner study: {study_name}")
        return study_name

    def get_suggestions(self, study_id: str, count: int = 1) -> List[Trial]:
        if study_id not in self.oracles:
            raise ValueError(f"Study {study_id} not found")
        
        data = self.oracles[study_id]
        oracle = data["oracle"]
        specs = data["specs"]
        
        trials = []
        for _ in range(count):
            # Create a HyperParameters object and populate it based on specs
            hp = keras_tuner.HyperParameters()
            for param_name, spec in specs.items():
                p_type = spec.get("type", "DOUBLE")
                
                if p_type == "DOUBLE":
                    min_val = spec.get("min_value", 0.0)
                    max_val = spec.get("max_value", 1.0)
                    sampling = "log" if spec.get("scale", "linear") == "log" else "linear"
                    hp.Float(param_name, min_value=min_val, max_value=max_val, sampling=sampling)
                    
                elif p_type == "INTEGER":
                    min_val = spec.get("min_value", 0)
                    max_val = spec.get("max_value", 100)
                    sampling = "log" if spec.get("scale", "linear") == "log" else "linear"
                    hp.Int(param_name, min_value=min_val, max_value=max_val, sampling=sampling)
                    
                elif p_type == "CATEGORICAL":
                    choices = spec.get("values", [])
                    hp.Choice(param_name, values=choices)

            # Ask Oracle for a trial
            # populate_space expects a trial_id usually, but create_trial generates one.
            # Actually oracle.create_trial(tuner_id) calls populate_space.
            keras_trial = oracle.create_trial(tuner_id="tuner0")
            
            # If the trial is newly created, its hyperparameters might be empty if we didn't pass hp.
            # Wait, Oracle.create_trial calls populate_space(trial_id).
            # We need to inject our search space into the oracle or the trial.
            # Keras Tuner's Oracle usually discovers space from the `build` function.
            # Here we are bypassing `build`.
            # We must manually update the trial's hyperparameters with our definition BEFORE populating?
            # No, populate_space generates values. But it needs to know the space.
            # The Oracle learns the space as it sees it.
            # So we should pass the `hp` object to `oracle.update_space(hp)` first.
            
            oracle.update_space(hp)
            
            # Now populate
            status = oracle.populate_space(keras_trial.trial_id)
            if status["status"] == "EXIT":
                break # No more trials
            
            # The values are in status['values'] or keras_trial.hyperparameters.values
            params = status["values"]
            if not params:
                 # If populate_space didn't return values (e.g. it's waiting), we might need to handle it.
                 # But for Bayesian, it should return.
                 params = keras_trial.hyperparameters.values
            
            trials.append(Trial(
                id=keras_trial.trial_id,
                parameters=params,
                metrics={},
                status="ACTIVE"
            ))
            
        return trials

    def report_metrics(self, trial_id: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # Keras Tuner Oracle doesn't strictly require intermediate reporting for the algorithm,
        # but we can use update_trial.
        pass

    def complete_trial(self, trial_id: str, final_measurement: Dict[str, float], infeasible: bool = False) -> None:
        # Find the oracle (assuming single study or we need to look it up)
        # We'll search
        target_oracle = None
        for data in self.oracles.values():
            oracle = data["oracle"]
            # We can't easily check if trial belongs to oracle without keeping track.
            # But we can try to end it.
            try:
                # Update metrics first
                oracle.update_trial(trial_id, metrics=final_measurement)
                # Then end trial
                oracle.end_trial(trial_id=trial_id, status="COMPLETED")
                target_oracle = oracle
                break
            except Exception as e:
                log.warning(f"Oracle rejected trial {trial_id}: {e}")
                continue
        
        if not target_oracle:
            log.warning(f"Could not find oracle for trial {trial_id}")
