from datetime import datetime
from typing import Dict, Any, List
import json
from pathlib import Path


class ExperimentTracker:
    """Track ML experiments"""

    def __init__(self, log_dir: str = "logs/experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, experiment_name: str, parameters: Dict[str, Any],
                       metrics: Dict[str, Any], tags: List[str], timestamp: str):
        """Log experiment"""
        experiment = {
            "experiment_name": experiment_name,
            "parameters": parameters,
            "metrics": metrics,
            "tags": tags,
            "timestamp": timestamp
        }

        # Save to file
        filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(experiment, f, indent=2)
        except Exception as e:
            print(f"Failed to log experiment: {e}")

    def get_experiments(self, experiment_name: str = None) -> List[Dict[str, Any]]:
        """Get experiments"""
        experiments = []

        for file in self.log_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    experiment = json.load(f)

                if experiment_name is None or experiment["experiment_name"] == experiment_name:
                    experiments.append(experiment)
            except Exception as e:
                print(f"Failed to read experiment file {file}: {e}")

        # Sort by timestamp
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return experiments

    def get_best_experiment(self, experiment_name: str, metric: str = "confidence") -> Dict[str, Any]:
        """Get best experiment by metric"""
        experiments = self.get_experiments(experiment_name)

        if not experiments:
            return {}

        best_experiment = max(experiments,
                              key=lambda x: x.get("metrics", {}).get(metric, 0))
        return best_experiment