"""
Model Manager for StaR
Provides improved model saving and loading with organized directory structure.
"""

import os
import json
import uuid
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class ModelManager:
    """
    Manages model saving and loading with improved organization.
    
    Directory structure:
    saved_models/
    ├── dataset_name/
    │   ├── method_name/
    │   │   ├── model_id_1/
    │   │   │   ├── model.pt
    │   │   │   ├── training_parameters.json
    │   │   │   ├── thresholds/
    │   │   │   │   ├── recon_threshold.npy
    │   │   │   │   ├── recon_mean.npy
    │   │   │   │   ├── recon_std.npy
    │   │   │   │   ├── lower_encoder.npy
    │   │   │   │   ├── upper_encoder.npy
    │   │   │   │   ├── us_mean_encoder.npy
    │   │   │   │   ├── us_std_encoder.npy
    │   │   │   │   ├── lower_decoder.npy
    │   │   │   │   ├── upper_decoder.npy
    │   │   │   │   ├── us_mean_decoder.npy
    │   │   │   │   └── us_std_decoder.npy
    │   │   │   └── metadata.json
    │   │   └── model_id_2/
    │   └── method_name_2/
    └── dataset_name_2/
    """
    
    def __init__(self, base_dir: str = "saved_models"):
        """
        Initialize ModelManager.
        
        Args:
            base_dir: Base directory for saving models
        """
        self.base_dir = os.path.join(os.getcwd(), base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_model_directory(self, dataset_name: str, method_name: str, 
                             model_params: Dict[str, Any]) -> Tuple[str, str]:
        """
        Create a unique directory for a model.
        
        Args:
            dataset_name: Name of the dataset
            method_name: Name of the Granger causality method
            model_params: Dictionary of model parameters
            
        Returns:
            Tuple of (model_id, full_path)
        """
        # Create dataset and method directories
        dataset_dir = os.path.join(self.base_dir, dataset_name)
        method_dir = os.path.join(dataset_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # Generate unique model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        model_id = f"{timestamp}_{unique_id}"
        
        # Create model directory
        model_dir = os.path.join(method_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create thresholds subdirectory
        thresholds_dir = os.path.join(model_dir, "thresholds")
        os.makedirs(thresholds_dir, exist_ok=True)
        
        # Save training parameters
        self._save_training_parameters(model_dir, model_params)
        
        return model_id, model_dir
    
    def _save_training_parameters(self, model_dir: str, params: Dict[str, Any]):
        """Save training parameters to JSON file."""
        # Create a clean copy of parameters for JSON serialization
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                clean_params[key] = value
            elif hasattr(value, '__name__'):  # For functions/classes
                clean_params[key] = str(value.__name__)
            elif torch.is_tensor(value):
                clean_params[key] = f"<Tensor: {value.shape}>"
            else:
                clean_params[key] = str(value)
        
        # Add metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "training_parameters": clean_params,
            "model_type": "StaR",
            "version": "1.0"
        }
        
        # Save to JSON file
        params_file = os.path.join(model_dir, "training_parameters.json")
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def save_model_state(self, model_dir: str, model_state_dict: Dict[str, Any]):
        """Save model state dictionary."""
        model_file = os.path.join(model_dir, "model.pt")
        torch.save(model_state_dict, model_file)
    
    def save_threshold(self, model_dir: str, threshold_name: str, threshold_value: np.ndarray):
        """Save a threshold array."""
        thresholds_dir = os.path.join(model_dir, "thresholds")
        threshold_file = os.path.join(thresholds_dir, f"{threshold_name}.npy")
        np.save(threshold_file, threshold_value)
    
    def load_model_state(self, model_dir: str, device: torch.device) -> Dict[str, Any]:
        """Load model state dictionary."""
        model_file = os.path.join(model_dir, "model.pt")
        return torch.load(model_file, map_location=device)
    
    def load_threshold(self, model_dir: str, threshold_name: str) -> np.ndarray:
        """Load a threshold array."""
        thresholds_dir = os.path.join(model_dir, "thresholds")
        threshold_file = os.path.join(thresholds_dir, f"{threshold_name}.npy")
        return np.load(threshold_file)
    
    def load_training_parameters(self, model_dir: str) -> Dict[str, Any]:
        """Load training parameters from JSON file."""
        params_file = os.path.join(model_dir, "training_parameters.json")
        with open(params_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_models(self, dataset_name: Optional[str] = None, 
                   method_name: Optional[str] = None) -> Dict[str, Any]:
        """
        List available models.
        
        Args:
            dataset_name: Filter by dataset name (optional)
            method_name: Filter by method name (optional)
            
        Returns:
            Dictionary with model information
        """
        models = {}
        
        # Determine which datasets to scan
        if dataset_name:
            datasets = [dataset_name] if os.path.exists(os.path.join(self.base_dir, dataset_name)) else []
        else:
            datasets = [d for d in os.listdir(self.base_dir) 
                       if os.path.isdir(os.path.join(self.base_dir, d))]
        
        for dataset in datasets:
            dataset_dir = os.path.join(self.base_dir, dataset)
            models[dataset] = {}
            
            # Determine which methods to scan
            if method_name:
                methods = [method_name] if os.path.exists(os.path.join(dataset_dir, method_name)) else []
            else:
                methods = [m for m in os.listdir(dataset_dir) 
                          if os.path.isdir(os.path.join(dataset_dir, m))]
            
            for method in methods:
                method_dir = os.path.join(dataset_dir, method)
                models[dataset][method] = []
                
                # List model instances
                model_instances = [m for m in os.listdir(method_dir) 
                                 if os.path.isdir(os.path.join(method_dir, m))]
                
                for model_id in model_instances:
                    model_dir = os.path.join(method_dir, model_id)
                    try:
                        params = self.load_training_parameters(model_dir)
                        models[dataset][method].append({
                            "model_id": model_id,
                            "path": model_dir,
                            "created_at": params.get("created_at", "Unknown"),
                            "parameters": params.get("training_parameters", {})
                        })
                    except Exception as e:
                        # Skip models with corrupted metadata
                        print(f"Warning: Could not load metadata for {model_dir}: {e}")
                        continue
        
        return models
    
    def find_latest_model(self, dataset_name: str, method_name: str) -> Optional[str]:
        """
        Find the latest model for a given dataset and method.
        
        Args:
            dataset_name: Name of the dataset
            method_name: Name of the method
            
        Returns:
            Path to the latest model directory, or None if not found
        """
        method_dir = os.path.join(self.base_dir, dataset_name, method_name)
        if not os.path.exists(method_dir):
            return None
        
        model_instances = [m for m in os.listdir(method_dir) 
                          if os.path.isdir(os.path.join(method_dir, m))]
        
        if not model_instances:
            return None
        
        # Sort by creation time (embedded in model_id)
        model_instances.sort(reverse=True)
        
        return os.path.join(method_dir, model_instances[0])
    
    def cleanup_old_models(self, dataset_name: str, method_name: str, keep_count: int = 5):
        """
        Clean up old models, keeping only the most recent ones.
        
        Args:
            dataset_name: Name of the dataset
            method_name: Name of the method
            keep_count: Number of models to keep
        """
        method_dir = os.path.join(self.base_dir, dataset_name, method_name)
        if not os.path.exists(method_dir):
            return
        
        model_instances = [m for m in os.listdir(method_dir) 
                          if os.path.isdir(os.path.join(method_dir, m))]
        
        if len(model_instances) <= keep_count:
            return
        
        # Sort by creation time and remove old ones
        model_instances.sort(reverse=True)
        models_to_remove = model_instances[keep_count:]
        
        import shutil
        for model_id in models_to_remove:
            model_path = os.path.join(method_dir, model_id)
            try:
                shutil.rmtree(model_path)
                print(f"Removed old model: {model_path}")
            except Exception as e:
                print(f"Warning: Could not remove {model_path}: {e}")
