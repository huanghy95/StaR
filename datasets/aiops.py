import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import random
from tqdm import tqdm


class AIOps:
    def __init__(self, options):
        """
        Initializes the AIOps Node Disk Fill dataset class.
        
        Args:
            options (dict): Configuration options for the dataset.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options.get('num_vars', 11)  # 11 nodes from variable_names.txt
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options.get('shuffle', True)
        self.fault_type = options.get('fault_type', 'node_disk_fill')
        self.time_series_length = options.get('time_series_length', 200)  # Reduced from 1000
        self.anomaly_ratio = options.get('anomaly_ratio', 0.8)
        self.training_size = options.get('training_size', 10)  # Reduced from 50
        self.testing_size = options.get('testing_size', 20)   # Reduced from 100
        
        # Load variable names
        self.variable_names = self._load_variable_names()
        
    def _load_variable_names(self):
        """Load variable names from the configuration file."""
        var_file = os.path.join(self.data_dir, self.fault_type, 'variable_names.txt')
        if os.path.exists(var_file):
            with open(var_file, 'r') as f:
                lines = f.readlines()
            variables = []
            for line in lines:
                if ':' in line:
                    # Extract variable name after the colon
                    var_name = line.split(':', 1)[1].strip()
                    variables.append(var_name)
            return variables
        else:
            # Default variable names if file doesn't exist
            return [f'node_{i}_disk_usage' for i in range(self.num_vars)]
    
    def _generate_synthetic_aiops_data(self):
        """
        Generate synthetic AIOps data based on the node disk fill scenario.
        This creates realistic time series data for filesystem usage rates.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        x_n_list = []  # Normal data
        x_ab_list = []  # Abnormal data
        label_list = []  # Labels
        
        total_samples = self.training_size + self.testing_size
        
        for sample_idx in tqdm(range(total_samples), desc="Generating AIOps data"):
            # Generate base time series for each node
            time_series = np.zeros((self.time_series_length, self.num_vars))
            labels = np.zeros((self.time_series_length, self.num_vars))
            
            # Initialize with baseline disk usage (20-40%)
            baseline_usage = np.random.uniform(0.2, 0.4, self.num_vars)
            time_series[0] = baseline_usage
            
            # Generate normal time series with some correlation between nodes
            for t in range(1, self.time_series_length):
                for node in range(self.num_vars):
                    # Add temporal correlation and some randomness
                    prev_value = time_series[t-1, node]
                    
                    # Slight trend and noise
                    trend = np.random.normal(0, 0.01)  # Small random trend
                    noise = np.random.normal(0, 0.02)  # Small noise
                    
                    # Correlation with other nodes (especially neighboring nodes)
                    correlation = 0
                    if node > 0:
                        correlation += 0.1 * (time_series[t-1, node-1] - baseline_usage[node-1])
                    if node < self.num_vars - 1:
                        correlation += 0.1 * (time_series[t-1, node+1] - baseline_usage[node+1])
                    
                    new_value = prev_value + trend + noise + correlation
                    time_series[t, node] = np.clip(new_value, 0.0, 1.0)
            
            # Create normal version
            x_normal = time_series.copy()
            
            # Create abnormal version with disk fill anomaly
            x_abnormal = time_series.copy()
            
            # Determine if this sample should have anomalies
            is_training = sample_idx < self.training_size
            
            # Ensure at least 50% of test samples have anomalies
            if not is_training:
                test_sample_idx = sample_idx - self.training_size
                # Force anomalies in at least half of test samples
                if test_sample_idx < self.testing_size // 2:
                    has_anomaly = True  # First half always has anomalies
                else:
                    has_anomaly = np.random.random() < self.anomaly_ratio
            else:
                has_anomaly = False  # No anomalies in training data
            
            if has_anomaly:
                # Select 1-3 nodes as root causes
                num_root_causes = np.random.randint(1, min(4, self.num_vars))
                root_cause_nodes = np.random.choice(self.num_vars, num_root_causes, replace=False)
                
                # Anomaly time window - ensure it's after window_size * 2 for proper evaluation
                # The model will slice labels with [window_size * 2:], so anomalies must be after that
                min_start = max(self.window_size * 2 + 10, int(0.6 * self.time_series_length))
                anomaly_start = np.random.randint(min_start, int(0.8 * self.time_series_length))
                max_duration = self.time_series_length - anomaly_start - 5  # Leave some buffer
                # Ensure minimum duration of 10 timesteps
                anomaly_duration = np.random.randint(10, max(11, max_duration))
                anomaly_end = min(anomaly_start + anomaly_duration, self.time_series_length)
                
                # Inject disk fill anomaly
                for t in range(anomaly_start, anomaly_end):
                    for root_node in root_cause_nodes:
                        # Gradual disk fill - exponential growth
                        if anomaly_duration > 0:
                            progress = (t - anomaly_start) / anomaly_duration
                        else:
                            progress = 1.0  # Full anomaly if duration is 0
                        anomaly_magnitude = 0.3 * (1 - np.exp(-3 * progress))  # Exponential growth
                        
                        x_abnormal[t, root_node] = min(1.0, x_abnormal[t, root_node] + anomaly_magnitude)
                        labels[t, root_node] = 1
                        
                        # Propagate effect to correlated nodes with delay
                        if t < anomaly_end - 1:
                            for other_node in range(self.num_vars):
                                if other_node != root_node:
                                    # Correlation-based propagation
                                    if abs(other_node - root_node) <= 2:  # Nearby nodes
                                        propagation_effect = 0.1 * anomaly_magnitude * np.random.uniform(0.5, 1.0)
                                        x_abnormal[t+1, other_node] = min(1.0, x_abnormal[t+1, other_node] + propagation_effect)
            
            x_n_list.append(x_normal)
            x_ab_list.append(x_abnormal)
            label_list.append(labels)
        
        return x_n_list, x_ab_list, label_list
    
    def generate_example(self):
        """
        Generate or load the AIOps dataset examples.
        """
        print(f"Generating AIOps {self.fault_type} dataset...")
        print(f"Variables: {len(self.variable_names)}")
        print(f"Variable names: {self.variable_names}")
        
        # Generate synthetic data
        x_n_list, x_ab_list, label_list = self._generate_synthetic_aiops_data()
        
        # Apply normalization
        scaler = StandardScaler()
        
        # Fit scaler on normal training data
        training_data = np.concatenate([x_n_list[i] for i in range(self.training_size)], axis=0)
        scaler.fit(training_data)
        
        # Transform all data
        x_n_list_scaled = []
        x_ab_list_scaled = []
        
        for i in range(len(x_n_list)):
            x_n_scaled = scaler.transform(x_n_list[i])
            x_ab_scaled = scaler.transform(x_ab_list[i])
            x_n_list_scaled.append(x_n_scaled)
            x_ab_list_scaled.append(x_ab_scaled)
        
        # Store in data_dict
        self.data_dict['x_n_list'] = np.array(x_n_list_scaled)
        self.data_dict['x_ab_list'] = np.array(x_ab_list_scaled)
        self.data_dict['label_list'] = np.array(label_list)
        
        # Shuffle if requested
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.arange(len(x_n_list_scaled))
            np.random.shuffle(indices)
            
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
            self.data_dict['x_ab_list'] = self.data_dict['x_ab_list'][indices]
            self.data_dict['label_list'] = self.data_dict['label_list'][indices]
        
        # Count anomalous samples for verification
        anomaly_count = 0
        anomaly_after_slice_count = 0
        slice_start = self.window_size * 2
        
        for i in range(len(label_list)):
            if np.any(label_list[i] == 1):
                anomaly_count += 1
            # Check if anomalies exist after the slice point (window_size * 2)
            if np.any(label_list[i][slice_start:] == 1):
                anomaly_after_slice_count += 1
        
        print(f"Generated {len(x_n_list_scaled)} samples")
        print(f"Training samples: {self.training_size}")
        print(f"Testing samples: {self.testing_size}")
        print(f"Anomalous samples: {anomaly_count}")
        print(f"Anomalous samples after slice (t>{slice_start}): {anomaly_after_slice_count}")
        print(f"Time series length: {self.time_series_length}")
        print(f"Number of variables: {self.num_vars}")
        print(f"Window size: {self.window_size}")
        
        # Verify we have anomalies in test data after slicing
        if anomaly_after_slice_count == 0:
            print("WARNING: No anomalous samples found after time slice! This will cause division by zero errors.")
            print(f"Anomalies must occur after timestep {slice_start} to be evaluated properly.")

    def save_data(self):
        """Save the generated data to disk."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Save data arrays
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])
        
        # Save metadata
        metadata = {
            'num_vars': self.num_vars,
            'variable_names': self.variable_names,
            'fault_type': self.fault_type,
            'time_series_length': self.time_series_length,
            'training_size': self.training_size,
            'testing_size': self.testing_size
        }
        
        import json
        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"AIOps dataset saved to: {self.data_dir}")

    def load_data(self):
        """Load existing data from disk."""
        try:
            # Load data arrays
            self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
            self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
            self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
            
            # Load metadata
            import json
            with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                
            print(f"AIOps dataset loaded from: {self.data_dir}")
            print(f"Fault type: {metadata['fault_type']}")
            print(f"Variables: {metadata['num_vars']}")
            print(f"Samples: {len(self.data_dict['x_n_list'])}")
            
        except FileNotFoundError as e:
            print(f"Data files not found: {e}")
            print("Please run with preprocessing_data=1 to generate the dataset first.")
            raise e

