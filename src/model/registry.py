import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry for versioning and management"""

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

        # Create default registry
        return {
            "models": [],
            "active_model": None,
            "last_updated": datetime.now().isoformat()
        }

    def _save_registry(self):
        """Save registry to file"""
        self.registry["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_model(self, model_path: str, metadata: Dict[str, Any]) -> str:
        """Register a new model in the registry"""
        model_id = hashlib.md5(
            f"{model_path}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Calculate file hash
        try:
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            file_hash = "unknown"

        model_entry = {
            "model_id": model_id,
            "path": model_path,
            "file_hash": file_hash,
            "metadata": metadata,
            "registered_at": datetime.now().isoformat(),
            "status": "staging"
        }

        # Add to registry
        self.registry["models"].append(model_entry)
        self._save_registry()

        logger.info(f"Registered model {model_id}")
        return model_id

    def promote_model(self, model_id: str, environment: str = "production"):
        """Promote a model to production"""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                model["status"] = environment
                model["promoted_at"] = datetime.now().isoformat()
                self.registry["active_model"] = model_id
                self._save_registry()
                logger.info(f"Promoted model {model_id} to {environment}")
                return True

        logger.error(f"Model {model_id} not found")
        return False

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Get the active production model"""
        if self.registry["active_model"]:
            for model in self.registry["models"]:
                if model["model_id"] == self.registry["active_model"]:
                    return model

        return None

    def get_model_history(self) -> List[Dict[str, Any]]:
        """Get model history"""
        return sorted(
            self.registry["models"],
            key=lambda x: x.get("registered_at", ""),
            reverse=True
        )

    def validate_model(self, model_path: str) -> bool:
        """Validate model file"""
        path = Path(model_path)

        if not path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        if file_size < 1024:  # Less than 1KB
            logger.error(f"Model file too small: {file_size} bytes")
            return False

        # Check file extension
        if path.suffix not in ['.pth', '.pt', '.bin']:
            logger.error(f"Invalid model file extension: {path.suffix}")
            return False

        return True

    def cleanup_old_models(self, keep_last: int = 5):
        """Clean up old model versions"""
        models = self.get_model_history()

        if len(models) <= keep_last:
            return

        # Keep the most recent models
        to_keep = models[:keep_last]
        to_remove = models[keep_last:]

        for model in to_remove:
            if model["status"] != "production":  # Don't remove production models
                try:
                    path = Path(model["path"])
                    if path.exists():
                        path.unlink()
                        logger.info(f"Removed old model: {model['model_id']}")
                except Exception as e:
                    logger.error(f"Failed to remove model {model['model_id']}: {e}")

        # Update registry
        self.registry["models"] = to_keep
        self._save_registry()