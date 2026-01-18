from datetime import datetime
from typing import Dict, Any, List
import json
from pathlib import Path


class PerformanceMonitor:
    """Monitor performance metrics"""

    def __init__(self, log_file: str = "logs/performance.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0,
            "recent_analyses": []
        }

    def record_analysis(self, analysis_id: str, confidence: float,
                        features: Dict[str, Any], timestamp: str):
        """Record analysis metrics"""
        self.metrics["total_analyses"] += 1
        self.metrics["successful_analyses"] += 1
        self.metrics["avg_confidence"] = (
                (self.metrics["avg_confidence"] * (self.metrics["total_analyses"] - 1) + confidence) /
                self.metrics["total_analyses"]
        )

        # Add to recent analyses
        analysis_record = {
            "id": analysis_id,
            "timestamp": timestamp,
            "confidence": confidence,
            "vessel_density": features.get("vessel_density", 0)
        }

        self.metrics["recent_analyses"].insert(0, analysis_record)
        self.metrics["recent_analyses"] = self.metrics["recent_analyses"][:10]  # Keep last 10

        # Log to file
        self._log_to_file(analysis_record)

    def _log_to_file(self, record: Dict[str, Any]):
        """Log to JSONL file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"Failed to log performance: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if self.metrics["total_analyses"] > 0:
            success_rate = (self.metrics["successful_analyses"] /
                            self.metrics["total_analyses"])
        else:
            success_rate = 0.0

        return {
            **self.metrics,
            "success_rate": success_rate
        }

    def get_recent_trends(self, window: int = 10) -> Dict[str, Any]:
        """Get recent trends"""
        recent = self.metrics["recent_analyses"][:window]

        if not recent:
            return {"count": 0, "avg_confidence": 0.0}

        confidences = [r["confidence"] for r in recent]
        densities = [r["vessel_density"] for r in recent]

        return {
            "count": len(recent),
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_density": sum(densities) / len(densities),
            "trend": "stable"  # Simplified trend calculation
        }