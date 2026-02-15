"""
Random Connection Service Dataset for StaR

This dataset implements a 5-service architecture with clean dependency flow:
- Service dependencies: service 2 <- service 1 <- service 3/4/5
- Edge services (3, 4, 5) can randomly connect TO service 1 ONLY
- Service 1 is always connected to service 2 (stable backbone)
- Root cause is randomly assigned to one or more of services [3, 4, 5]
- Problems occur when root cause services connect to service 1
- NO inter-connections between edge services (clean causal structure)

This design is particularly favorable to TGN models because:
1. Dynamic graph structure that changes at each time step
2. Memory effects: past connection patterns influence current behavior
3. Complex temporal dependencies that require tracking connection history
4. Realistic dependency flow from edge services to core infrastructure
5. Clean causal interpretation without confounding inter-connections
"""

import numpy as np
import random
import os
from tqdm import tqdm


class RandomConnectionService:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.num_vars = 5  # 5 services
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.noise_scale = options['noise_scale']
        self.dependent_features = options['dependent_features']
        
        # Service indices
        self.SERVICE_1 = 0  # Core service 1 (always connected to service 2)
        self.SERVICE_2 = 1  # Core service 2 (always connected to service 1)
        self.SERVICE_3 = 2  # Variable service 3
        self.SERVICE_4 = 3  # Variable service 4
        self.SERVICE_5 = 4  # Variable service 5
        
        # Connection parameters
        self.base_connection_prob = options.get('base_connection_prob', 0.65)  # Base probability for random connections
        self.connection_memory_decay = 0.9  # How much past connections influence current state
        self.full_connection_threshold = 0.8  # Threshold for "all connected" state
        
        # Error parameters (TGN advantages)
        self.error_buildup_rate = 0.1
        
        # Root cause assignment strategy (simplified to single_variable only)
        self.root_cause_strategy = 'single_variable'
        self.error_decay_rate = 0.05   # How fast errors decay when not all connected
        self.memory_window = 20        # How far back to look for connection patterns
        
        # Controlled root cause assignment (for systematic testing)
        self.fixed_root_cause = options.get('fixed_root_cause', None)  # None, 3, 4, or 5
        self.training_root_cause = options.get('training_root_cause', None)  # Specific for training
        self.testing_root_cause = options.get('testing_root_cause', None)   # Specific for testing
        
        self.generate_base_causal_structure()

    def generate_base_causal_structure(self):
        """Generate base causal structure with correct dependency flow: 2 <- 1 <- 3/4/5."""
        # Base structure: Service 2 <- Service 1 (always connected)
        # Edge services 3,4,5 can connect TO service 1 (creating dependencies)
        self.base_causal_struct = np.array([
            [0.8, 0.0, 0.0, 0.0, 0.0],  # Service 1: autoregressive only (receives from 3,4,5)
            [0.6, 0.8, 0.0, 0.0, 0.0],  # Service 2: autoregressive + depends on Service 1
            [0.0, 0.0, 0.7, 0.0, 0.0],  # Service 3: mostly independent (can connect to 1)
            [0.0, 0.0, 0.0, 0.7, 0.0],  # Service 4: mostly independent (can connect to 1)
            [0.0, 0.0, 0.0, 0.0, 0.7]   # Service 5: mostly independent (can connect to 1)
        ])
        
        # Store for compatibility
        self.data_dict['causal_struct'] = (self.base_causal_struct > 0).astype(float)

    def generate_connection_pattern(self, t, connection_history, sample_idx):
        """Generate dynamic connection pattern for time t with correct dependency flow."""
        # Use deterministic randomness based on time and sample
        np.random.seed(self.seed + sample_idx * 10000 + t)
        
        # Service 1 always depends on Service 2 (2 -> 1)
        connections = np.zeros((self.num_vars, self.num_vars))
        connections[self.SERVICE_2, self.SERVICE_1] = 1.0  # Service 2 influences Service 1
        
        # Compute connection probabilities for edge services 3, 4, 5 TO service 1
        # Probability influenced by recent connection history (TGN memory advantage)
        edge_services = [self.SERVICE_3, self.SERVICE_4, self.SERVICE_5]
        
        for svc in edge_services:
            # Base probability
            prob = self.base_connection_prob
            
            # Adjust based on recent connection history
            if len(connection_history) > 0:
                recent_connections = [conn_matrix[svc, self.SERVICE_1] 
                                    for conn_matrix in connection_history[-self.memory_window:]]
                if recent_connections:
                    recent_avg = np.mean(recent_connections)
                    # Services that were recently connected are more likely to connect again
                    prob += 0.15 * recent_avg
                    # But also add some anti-correlation to create interesting patterns
                    prob -= 0.05 * (recent_avg ** 2)
            
            # Ensure probability is in valid range
            prob = np.clip(prob, 0.4, 0.85)  # Higher minimum to ensure frequent connections
            
            # Generate connections: edge service connects TO service 1
            if np.random.random() < prob:
                # Edge service influences service 1 (correct dependency direction)
                connections[svc, self.SERVICE_1] = 1.0
                
                # REMOVED: No inter-connections between edge services (3, 4, 5)
                # This ensures clean causal structure: only 3/4/5 -> 1 -> 2 connections
        
        return connections

    def has_problematic_connections(self, connection_matrix, root_cause_services):
        """Check if any root cause services are connected to service 1 (problem condition)."""
        # Problem exists when any root cause service connects to service 1
        for svc in root_cause_services:
            if connection_matrix[svc, self.SERVICE_1] > 0:
                return True
        return False
    
    def count_edge_connections(self, connection_matrix):
        """Count how many edge services are connected to service 1."""
        connected_count = 0
        for svc in [self.SERVICE_3, self.SERVICE_4, self.SERVICE_5]:
            if connection_matrix[svc, self.SERVICE_1] > 0:
                connected_count += 1
        return connected_count

    def compute_service_load(self, service_values, connection_matrix, service_idx):
        """Compute load on a service based on incoming connections (dependencies)."""
        load = 0.0
        
        # Load from services that this service depends on (incoming connections)
        for other_svc in range(self.num_vars):
            if connection_matrix[other_svc, service_idx] > 0:
                connection_strength = connection_matrix[other_svc, service_idx]
                load += connection_strength * service_values[other_svc]
        
        return load

    def compute_error_state(self, connection_history, error_history, root_cause_services):
        """Compute system error state based on root cause connections (TGN memory advantage)."""
        if len(connection_history) == 0:
            return 0.0
        
        current_error = error_history[-1] if error_history else 0.0
        
        # Check if any root cause services are currently connected to service 1
        if self.has_problematic_connections(connection_history[-1], root_cause_services):
            # Error builds up when root cause services connect to service 1
            current_error += self.error_buildup_rate
            
            # Non-linear amplification if this has been going on for a while
            recent_problematic_connections = sum(1 for conn_matrix in connection_history[-10:] 
                                               if self.has_problematic_connections(conn_matrix, root_cause_services))
            if recent_problematic_connections > 7:  # More than 70% of recent time steps
                current_error += 0.05 * (recent_problematic_connections - 7)  # Accelerated buildup
        else:
            # Error decays when root cause services are not connected
            current_error *= (1.0 - self.error_decay_rate)
        
        return np.clip(current_error, 0.0, 2.0)

    def generate_example(self):
        """Generate random connection service examples."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []
        timing_info = []
        connection_history_list = []  # Store connection matrices for GAAC metric

        for i in tqdm(range(self.n), desc="Generating Random Connection Service Data"):
            # Determine if this is a training sample
            is_training = i < self.options['training_size']
            # Generate noise
            if self.dependent_features == 1:
                # Correlated noise reflecting service dependencies
                covariance_matrix = np.eye(self.num_vars) * 0.04
                # Core services are more correlated
                covariance_matrix[self.SERVICE_1, self.SERVICE_2] = 0.02
                covariance_matrix[self.SERVICE_2, self.SERVICE_1] = 0.02
                
                mean = np.zeros(self.num_vars)
                eps = self.noise_scale * np.random.multivariate_normal(mean, covariance_matrix, size=self.t)
            else:
                eps = self.noise_scale * np.random.randn(self.t, self.num_vars)

            eps_normal = eps.copy()
            eps_anom = eps.copy()

            # Initialize time series
            x = np.zeros((self.t, self.num_vars))
            x_ab = np.zeros((self.t, self.num_vars))
            
            # Initialize with baseline service values
            x[0, :] = np.array([0.3, 0.3, 0.2, 0.2, 0.2]) + np.random.normal(0.0, 0.05, self.num_vars)
            x_ab[0, :] = x[0, :].copy()

            # Determine root cause services for this sample first
            # Check for controlled root cause assignment
            if self.fixed_root_cause is not None:
                # Fixed root cause for all samples
                root_cause_services = [self.fixed_root_cause]
            elif is_training and self.training_root_cause is not None:
                # Specific root cause for training
                root_cause_services = [self.training_root_cause]
            elif not is_training and self.testing_root_cause is not None:
                # Specific root cause for testing
                root_cause_services = [self.testing_root_cause]
            else:
                # Random selection from edge services (can be multiple)
                edge_candidates = [self.SERVICE_3, self.SERVICE_4, self.SERVICE_5]
                num_root_causes = np.random.choice([1, 2], p=[0.7, 0.3])  # Usually 1, sometimes 2
                root_cause_services = list(np.random.choice(edge_candidates, size=num_root_causes, replace=False))

            # Track connection patterns and errors over time
            connection_history = []
            error_history = []
            problematic_connection_times = []

            # Generate time series with dynamic connections
            for t in range(1, self.t):
                # Generate connection pattern for this time step
                connection_matrix = self.generate_connection_pattern(t, connection_history, i)
                connection_history.append(connection_matrix)
                
                # Compute system error state based on root cause connections
                error_state = self.compute_error_state(connection_history, error_history, root_cause_services)
                error_history.append(error_state)
                
                # Track when root cause services connect to service 1 (problematic state)
                if self.has_problematic_connections(connection_matrix, root_cause_services):
                    problematic_connection_times.append(int(t))
                
                # Service dynamics based on connections and loads
                
                # Service 1: Core service, affected by connections and error state
                load_1 = self.compute_service_load(x[t-1, :], connection_matrix, self.SERVICE_1)
                x[t, self.SERVICE_1] = (0.8 * x[t-1, self.SERVICE_1] + 
                                       0.1 * load_1 + 
                                       0.05 * error_state +  # Errors affect core services
                                       eps_normal[t, self.SERVICE_1])
                
                # Service 2: Core service, similar to service 1 but slightly different dynamics
                load_2 = self.compute_service_load(x[t-1, :], connection_matrix, self.SERVICE_2)
                x[t, self.SERVICE_2] = (0.8 * x[t-1, self.SERVICE_2] + 
                                       0.12 * load_2 + 
                                       0.03 * error_state +
                                       eps_normal[t, self.SERVICE_2])
                
                # Edge services (3, 4, 5): Behavior depends on whether they connect to service 1
                for svc in [self.SERVICE_3, self.SERVICE_4, self.SERVICE_5]:
                    load = self.compute_service_load(x[t-1, :], connection_matrix, svc)
                    
                    if connection_matrix[svc, self.SERVICE_1] > 0:  # This service connects TO service 1
                        # When connected to service 1, edge service is more active
                        base_activity = 0.6 * x[t-1, svc]
                        connection_boost = 0.2 * load
                        
                        # If this is a root cause service, it contributes to system stress
                        if svc in root_cause_services:
                            stress_factor = 0.15 * error_state
                            # Non-linear interaction when error state is high (TGN+KGC advantage)
                            if error_state > 0.5:
                                stress_factor *= (1.0 + np.tanh(2.0 * error_state))
                        else:
                            stress_factor = 0.05 * error_state
                        
                        x[t, svc] = (base_activity + connection_boost + stress_factor + 
                                   eps_normal[t, svc])
                    else:
                        # When disconnected from service 1, edge service runs independently
                        x[t, svc] = (0.7 * x[t-1, svc] + 
                                   0.05 * load +  # Minimal background load
                                   eps_normal[t, svc])

            # Generate anomalous version
            x_ab = x.copy()
            
            # Determine anomaly timing based on problematic connections
            if len(problematic_connection_times) > 10:  # Only if there are enough problematic connections
                # Anomaly occurs during a period of sustained problematic connections
                sustained_periods = []
                current_period = [problematic_connection_times[0]]
                
                for idx in range(1, len(problematic_connection_times)):
                    if problematic_connection_times[idx] - problematic_connection_times[idx-1] <= 3:  # Within 3 time steps
                        current_period.append(problematic_connection_times[idx])
                    else:
                        if len(current_period) >= 5:  # At least 5 consecutive problematic connections
                            sustained_periods.append(current_period)
                        current_period = [problematic_connection_times[idx]]
                
                if len(current_period) >= 5:
                    sustained_periods.append(current_period)
                
                if sustained_periods:
                    # Choose the longest sustained period for anomaly
                    anomaly_period = max(sustained_periods, key=len)
                    anomaly_start = anomaly_period[len(anomaly_period)//2]  # Middle of the period
                    anomaly_length = min(self.adlength, len(anomaly_period)//2)
                    
                    # Inject anomaly with realistic cascading effects
                    for t_idx in range(anomaly_start, min(anomaly_start + anomaly_length, self.t)):
                        # Root cause services get the highest anomaly
                        for root_svc in root_cause_services:
                            x_ab[t_idx, root_svc] += 1.0 * self.mul
                        
                        # Secondary effects on core services (cascading failure)
                        # Service 1 gets moderate impact (receives from root causes)
                        x_ab[t_idx, self.SERVICE_1] += 0.4 * self.mul
                        # Service 2 gets less impact (depends on service 1)
                        x_ab[t_idx, self.SERVICE_2] += 0.2 * self.mul
                        
                        # Other edge services get minimal impact
                        for svc in [self.SERVICE_3, self.SERVICE_4, self.SERVICE_5]:
                            if svc not in root_cause_services:
                                x_ab[t_idx, svc] += 0.1 * self.mul
                else:
                    anomaly_start = int(0.7 * self.t)
                    anomaly_length = self.adlength
            else:
                # Fallback: anomaly in the latter part
                anomaly_start = int(0.7 * self.t)
                anomaly_length = self.adlength

            # Create labels with realistic root cause assignment
            temp_label = np.zeros((self.t, self.num_vars))
            if 'anomaly_start' in locals():
                anomaly_end = min(anomaly_start + anomaly_length, self.t)
                
                # Mark all root cause services in the labels
                for root_svc in root_cause_services:
                    temp_label[anomaly_start:anomaly_end, root_svc] = 1

            # Clip values to reasonable ranges
            x = np.clip(x, 0, 3)
            x_ab = np.clip(x_ab, 0, 6)

            x_n_list.append(x)
            x_ab_list.append(x_ab)
            eps_n_list.append(eps_normal)
            eps_ab_list.append(eps_anom)
            label_list.append(temp_label)
            
            # Store connection history as numpy array for GAAC metric
            # Shape: (T-1, num_vars, num_vars) - one connection matrix per timestep
            connection_history_array = np.stack(connection_history) if connection_history else np.zeros((self.t-1, self.num_vars, self.num_vars))
            connection_history_list.append(connection_history_array)

            # Store timing and connection info
            timing_info.append({
                'problematic_connection_times': problematic_connection_times,
                'anomaly_start': int(anomaly_start) if 'anomaly_start' in locals() else None,
                'anomaly_end': int(anomaly_start + anomaly_length) if 'anomaly_start' in locals() else None,
                'connection_patterns': len(connection_history),
                'max_error_state': float(max(error_history)) if error_history else 0.0,
                'root_cause_strategy': 'multiple_variable',
                'root_cause_services': [int(svc) for svc in root_cause_services],
                'num_root_causes': len(root_cause_services)
            })

        # Store all data
        self.data_dict['x_n_list'] = x_n_list
        self.data_dict['x_ab_list'] = x_ab_list
        self.data_dict['eps_n_list'] = eps_n_list
        self.data_dict['eps_ab_list'] = eps_ab_list
        self.data_dict['label_list'] = label_list
        self.data_dict['timing_info'] = timing_info
        self.data_dict['connection_history_list'] = connection_history_list

    def save_data(self):
        """Save the generated data to disk."""
        if not hasattr(self, 'data_dict') or not self.data_dict:
            print("No data to save. Please run generate_example() first.")
            return
            
        os.makedirs(self.data_dir, exist_ok=True)
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list.npy'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list.npy'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct.npy'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'connection_history_list.npy'), self.data_dict['connection_history_list'])
        
        # Save timing information as JSON
        import json
        with open(os.path.join(self.data_dir, 'timing_info.json'), 'w') as f:
            json.dump(self.data_dict['timing_info'], f, indent=2)
            
        print(f"Random Connection Service dataset saved to: {self.data_dir}")
        print(f"  - Connection history saved for GAAC metric evaluation")

    def load_data(self):
        """Load existing data from disk."""
        try:
            self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=True)
            self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
            self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'), allow_pickle=True)
            self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'), allow_pickle=True)
            self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)
            self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
            
            # Load connection history for GAAC metric (may not exist in older datasets)
            connection_history_path = os.path.join(self.data_dir, 'connection_history_list.npy')
            if os.path.exists(connection_history_path):
                self.data_dict['connection_history_list'] = np.load(connection_history_path, allow_pickle=True)
                print(f"  - Connection history loaded for GAAC metric")
            else:
                print(f"  - Warning: connection_history_list.npy not found. GAAC metric will not be available.")
                print(f"  - Please regenerate the dataset with preprocessing_data=1 to enable GAAC.")
                self.data_dict['connection_history_list'] = None
            
            # Load timing information
            import json
            with open(os.path.join(self.data_dir, 'timing_info.json'), 'r') as f:
                self.data_dict['timing_info'] = json.load(f)
                
            print(f"Random Connection Service dataset loaded from: {self.data_dir}")
            
        except FileNotFoundError as e:
            print(f"Data files not found: {e}")
            print("Please run generate_example() first to create the dataset.")

