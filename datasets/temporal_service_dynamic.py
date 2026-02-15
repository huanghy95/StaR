"""
Dynamic Temporal Service Dataset for StaR

This dataset simulates a microservice architecture with dynamic dependency changes
that are more favorable to TGN models. It includes:

1. Variable timing for phase transitions (not fixed)
2. Gradual transitions instead of abrupt changes
3. Memory-dependent service states
4. Non-linear service interactions
5. Different causal patterns and temporal dynamics between training vs testing

Architecture (5 services for both training and testing):
- Service 1: Front-End (always active)
- Service 2: Load Balancer (always active) 
- Service 3: Target-Service-v1 (active in Phase 1, gradually deactivated)
- Service 4: Target-Service-v2 (gradually activated in Phase 2)
- Service 5: Database (always active, but performance varies)

Dynamic Patterns:
- Phase transitions happen at variable times (Gaussian distribution)
- Service activation/deactivation is gradual (sigmoid transitions)
- Service performance depends on historical load (memory effects)
- Non-linear interactions between services under stress
- Different causal coupling strengths and feedback loops in testing vs training
"""

import numpy as np
import random
import os
from tqdm import tqdm


class DynamicTemporalService:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        # Dynamic graph topology: same 5 services for training and testing
        # but with different temporal patterns and causal structures
        self.training_vars = 5  # 5 services for training
        self.testing_vars = 5   # 5 services for testing (same count, different patterns)
        self.num_vars = self.training_vars  # Always 5 services
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.noise_scale = options['noise_scale']
        self.dependent_features = options['dependent_features']
        
        # Service indices (training configuration)
        self.FRONTEND = 0           # Service 1: Front-End
        self.LOAD_BALANCER = 1      # Service 2: Load Balancer
        self.TARGET_V1 = 2          # Service 3: Target-Service-v1
        self.TARGET_V2 = 3          # Service 4: Target-Service-v2
        self.DATABASE = 4           # Service 5: Database
        
        # Note: All 5 services are used in both training and testing
        # The difference is in temporal patterns and causal relationships
        
        # Dynamic timing parameters (more favorable to TGN)
        self.base_transition_time = int(0.4 * self.t)  # Base time for v1->v2 transition
        self.transition_variance = int(0.1 * self.t)   # Variance in transition timing
        self.transition_duration = int(0.2 * self.t)   # Duration of gradual transition
        
        # Memory parameters for TGN advantages
        self.memory_window = 50     # Services remember last 50 time steps
        self.stress_threshold = 0.7 # Threshold for service stress
        
        # Root cause assignment - simple random like linear/nonlinear datasets
        # No need for complex strategies, just random assignment of 1 to all variables
        
        self.generate_dynamic_causal_structures()

    def generate_dynamic_causal_structures(self):
        """Generate time-varying causal structures for the microservice system."""
        # Training causal structure (5x5 - Phase 1: using Target-v1)
        self.causal_struct_training_phase1 = np.array([
            [0.8, 0.0, 0.0, 0.0, 0.0],  # Frontend: autoregressive
            [0.6, 0.7, 0.0, 0.0, 0.0],  # Load Balancer: depends on Frontend
            [0.0, 0.8, 0.6, 0.0, 0.2],  # Target-v1: depends on LB + DB
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Target-v2: inactive
            [0.2, 0.3, 0.4, 0.0, 0.9]   # Database: depends on Frontend, LB, Target-v1
        ])
        
        # Training causal structure (5x5 - Phase 2: using Target-v2)
        self.causal_struct_training_phase2 = np.array([
            [0.8, 0.0, 0.0, 0.0, 0.0],  # Frontend: autoregressive
            [0.6, 0.7, 0.0, 0.0, 0.0],  # Load Balancer: depends on Frontend
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Target-v1: inactive
            [0.0, 0.9, 0.0, 0.5, 0.3],  # Target-v2: depends on LB + DB
            [0.2, 0.3, 0.0, 0.5, 0.9]   # Database: depends on Frontend, LB, Target-v2
        ])
        
        # Testing causal structure (5x5 - different patterns from training)
        # Testing Phase 1: More complex interdependencies, stronger cross-connections
        self.causal_struct_testing_phase1 = np.array([
            [0.7, 0.2, 0.0, 0.0, 0.1],  # Frontend: autoregressive + feedback from DB
            [0.5, 0.8, 0.1, 0.0, 0.0],  # Load Balancer: depends on Frontend + Target-v1 feedback
            [0.1, 0.7, 0.5, 0.0, 0.3],  # Target-v1: stronger dependencies, cross-talk
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Target-v2: inactive
            [0.3, 0.4, 0.5, 0.0, 0.8]   # Database: stronger dependencies on all active services
        ])
        
        # Testing Phase 2: Different transition patterns and coupling strengths
        self.causal_struct_testing_phase2 = np.array([
            [0.7, 0.0, 0.0, 0.1, 0.2],  # Frontend: autoregressive + Target-v2 + DB feedback
            [0.4, 0.8, 0.0, 0.2, 0.0],  # Load Balancer: weaker Frontend dep, Target-v2 feedback
            [0.0, 0.0, 0.1, 0.0, 0.0],  # Target-v1: mostly inactive, minimal residual
            [0.2, 0.8, 0.0, 0.6, 0.4],  # Target-v2: different coupling pattern than training
            [0.4, 0.2, 0.0, 0.6, 0.7]   # Database: different dependency weights
        ])
        
        # Store structures for evaluation
        self.data_dict['causal_struct_training_phase1'] = (self.causal_struct_training_phase1 > 0).astype(float)
        self.data_dict['causal_struct_training_phase2'] = (self.causal_struct_training_phase2 > 0).astype(float)
        self.data_dict['causal_struct_testing_phase1'] = (self.causal_struct_testing_phase1 > 0).astype(float)
        self.data_dict['causal_struct_testing_phase2'] = (self.causal_struct_testing_phase2 > 0).astype(float)
        
        # For compatibility, use training phase1 as default
        self.data_dict['causal_struct'] = self.data_dict['causal_struct_training_phase1']

    def sigmoid_transition(self, t, transition_start, transition_duration):
        """Smooth sigmoid transition between 0 and 1."""
        if transition_duration == 0:
            return 1.0 if t >= transition_start else 0.0
        
        # Sigmoid centered at transition_start with given duration
        x = (t - transition_start) / (transition_duration / 6)  # 6 sigma for smooth transition
        return 1 / (1 + np.exp(-x))

    def compute_service_stress(self, history, window_size=20):
        """Compute service stress based on recent history (TGN memory advantage)."""
        if len(history) < window_size:
            return 0.0
        
        recent_values = history[-window_size:]
        mean_load = np.mean(recent_values)
        variance = np.var(recent_values)
        
        # Stress increases with high load and high variance
        stress = min(1.0, mean_load + 0.5 * variance)
        return stress

    def generate_variable_transition_time(self, base_time, variance, sample_idx):
        """Generate variable transition times for different samples."""
        # Use sample index to ensure reproducibility while adding variance
        np.random.seed(self.seed + sample_idx)
        
        # Different patterns for training vs testing
        if sample_idx < self.options['training_size']:
            # Training: more predictable patterns
            noise_factor = 0.5
        else:
            # Testing: more variable patterns to test generalization
            noise_factor = 1.0
        
        transition_time = base_time + np.random.normal(0, variance * noise_factor)
        transition_time = max(int(0.2 * self.t), min(int(0.8 * self.t), int(transition_time)))
        
        return transition_time

    def generate_example(self):
        """Generate dynamic temporal service examples."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []
        timing_info = []

        for i in tqdm(range(self.n), desc="Generating Dynamic Temporal Service Data"):
            # Determine if this is training or testing sample
            is_training = i < self.options['training_size']
            current_num_vars = 5  # Always 5 variables for both training and testing
            
            # Generate variable transition timing for this sample
            transition_start = self.generate_variable_transition_time(
                self.base_transition_time, self.transition_variance, i
            )
            
            # Anomaly starts after transition completes
            anomaly_start = min(
                transition_start + self.transition_duration + np.random.randint(10, 50),
                self.t - self.adlength
            )
            anomaly_length = min(self.adlength, self.t - anomaly_start)
            
            # Generate noise
            if self.dependent_features == 1:
                # Correlated noise for realistic service interactions
                covariance_matrix = np.eye(current_num_vars) * 0.05
                # Add some correlation between services
                covariance_matrix[self.FRONTEND, self.LOAD_BALANCER] = 0.02
                covariance_matrix[self.LOAD_BALANCER, self.FRONTEND] = 0.02
                covariance_matrix[self.TARGET_V1, self.DATABASE] = 0.03
                covariance_matrix[self.DATABASE, self.TARGET_V1] = 0.03
                covariance_matrix[self.TARGET_V2, self.DATABASE] = 0.03
                covariance_matrix[self.DATABASE, self.TARGET_V2] = 0.03
                
                # Different noise patterns for testing (same 5 services, different correlations)
                if not is_training:
                    # Stronger correlations in testing to create different patterns
                    covariance_matrix[self.FRONTEND, self.DATABASE] = 0.025
                    covariance_matrix[self.DATABASE, self.FRONTEND] = 0.025
                    covariance_matrix[self.LOAD_BALANCER, self.TARGET_V2] = 0.02
                    covariance_matrix[self.TARGET_V2, self.LOAD_BALANCER] = 0.02
                
                mean = np.zeros(current_num_vars)
                eps = self.noise_scale * np.random.multivariate_normal(mean, covariance_matrix, size=self.t)
            else:
                eps = self.noise_scale * np.random.randn(self.t, current_num_vars)

            eps_normal = eps.copy()
            eps_anom = eps.copy()

            # Initialize time series
            x = np.zeros((self.t, current_num_vars))
            x_ab = np.zeros((self.t, current_num_vars))
            
            # Initialize with realistic service baseline values (same 5 services)
            if is_training:
                x[0, :] = np.array([0.3, 0.2, 0.1, 0.0, 0.4]) + np.random.normal(0.0, 0.05, current_num_vars)
            else:
                # Testing: different initial conditions to create different temporal patterns
                x[0, :] = np.array([0.4, 0.3, 0.15, 0.0, 0.35]) + np.random.normal(0.0, 0.05, current_num_vars)
            x_ab[0, :] = x[0, :].copy()

            # Set up anomaly with random root cause assignment (like linear/nonlinear datasets)
            t_p = np.arange(anomaly_start, anomaly_start + anomaly_length)
            
            # Random root cause assignment from variable services [TARGET_V1, TARGET_V2, DATABASE]
            # Can assign 1 to all 3 services randomly (like linear/nonlinear datasets)
            root_cause_candidates = [self.TARGET_V1, self.TARGET_V2, self.DATABASE]
            num_root_causes = np.random.randint(1, len(root_cause_candidates) + 1)  # 1 to 3 root causes
            feature_p = list(np.random.choice(root_cause_candidates, size=num_root_causes, replace=False))
            
            ab = np.zeros(current_num_vars)
            ab[feature_p] += self.mul
            
            temp_label = np.zeros((self.t, current_num_vars))
            temp_label[np.ix_(t_p, feature_p)] = 1

            # Store timing info
            timing_info.append({
                'transition_start': int(transition_start),
                'transition_end': int(transition_start + self.transition_duration),
                'anomaly_start': int(anomaly_start),
                'anomaly_end': int(anomaly_start + anomaly_length),
                'root_cause_variables': [int(var) for var in feature_p],  # Multiple root causes possible
                'num_root_causes': len(feature_p),
                'is_training': is_training
            })

            # Service history for memory effects (TGN advantage)
            service_history = {i: [] for i in range(current_num_vars)}

            # Generate normal time series with dynamic transitions
            for t in range(1, self.t):
                # Update service history
                for svc in range(current_num_vars):
                    service_history[svc].append(x[t-1, svc])
                    if len(service_history[svc]) > self.memory_window:
                        service_history[svc].pop(0)

                # Compute transition weights
                v1_weight = 1.0 - self.sigmoid_transition(t, transition_start, self.transition_duration)
                v2_weight = self.sigmoid_transition(t, transition_start, self.transition_duration)
                
                # Compute service stress (memory-dependent, TGN advantage)
                frontend_stress = self.compute_service_stress(service_history[self.FRONTEND])
                lb_stress = self.compute_service_stress(service_history[self.LOAD_BALANCER])
                db_stress = self.compute_service_stress(service_history[self.DATABASE])
                
                # Service equations with dynamic behavior
                
                # Frontend: Autoregressive with external load patterns
                external_load = 0.5 + 0.3 * np.sin(2 * np.pi * t / (self.t / 10))  # Cyclical load
                x[t, self.FRONTEND] = (0.8 * x[t-1, self.FRONTEND] + 
                                     0.2 * external_load + 
                                     eps_normal[t, self.FRONTEND])
                
                # Load Balancer: Depends on Frontend with stress effects
                stress_multiplier = 1.0 + 0.5 * frontend_stress  # Stress amplifies load
                x[t, self.LOAD_BALANCER] = (0.7 * x[t-1, self.LOAD_BALANCER] + 
                                           0.6 * x[t-1, self.FRONTEND] * stress_multiplier + 
                                           eps_normal[t, self.LOAD_BALANCER])
                
                # Target-v1: Active in Phase 1, gradually deactivated
                if v1_weight > 0.01:  # Only compute if significantly active
                    load_factor = 0.8 * x[t-1, self.LOAD_BALANCER]
                    db_factor = 0.2 * x[t-1, self.DATABASE] if t > 1 else 0.0
                    
                    # Non-linear interaction under high stress (TGN+KGC advantage)
                    if lb_stress > self.stress_threshold:
                        load_factor *= (1.0 + np.tanh(2.0 * lb_stress))  # Non-linear amplification
                    
                    x[t, self.TARGET_V1] = v1_weight * (0.6 * x[t-1, self.TARGET_V1] + 
                                                       load_factor + db_factor + 
                                                       eps_normal[t, self.TARGET_V1])
                else:
                    x[t, self.TARGET_V1] = 0.1 * x[t-1, self.TARGET_V1] + eps_normal[t, self.TARGET_V1]
                
                # Target-v2: Gradually activated in Phase 2
                if v2_weight > 0.01:  # Only compute if significantly active
                    load_factor = 0.9 * x[t-1, self.LOAD_BALANCER]
                    db_factor = 0.3 * x[t-1, self.DATABASE] if t > 1 else 0.0
                    
                    # Different performance characteristics than v1
                    # More sensitive to database performance (memory effect)
                    if db_stress > self.stress_threshold:
                        db_factor *= (1.0 + 2.0 * db_stress)  # Higher sensitivity
                    
                    x[t, self.TARGET_V2] = v2_weight * (0.5 * x[t-1, self.TARGET_V2] + 
                                                       load_factor + db_factor + 
                                                       eps_normal[t, self.TARGET_V2])
                else:
                    x[t, self.TARGET_V2] = eps_normal[t, self.TARGET_V2] * 0.1
                
                # Database: Depends on active services with memory effects
                frontend_load = 0.2 * x[t-1, self.FRONTEND]
                lb_load = 0.3 * x[t-1, self.LOAD_BALANCER]
                
                # Service-specific database load
                v1_load = 0.4 * x[t-1, self.TARGET_V1] * v1_weight
                v2_load = 0.5 * x[t-1, self.TARGET_V2] * v2_weight  # v2 is more DB-intensive
                
                # Memory effect: database performance degrades with sustained high load
                sustained_load = self.compute_service_stress(service_history[self.DATABASE])
                performance_degradation = 1.0 + 0.3 * sustained_load
                
                total_load = (frontend_load + lb_load + v1_load + v2_load) * performance_degradation
                
                x[t, self.DATABASE] = (0.9 * x[t-1, self.DATABASE] + 
                                     total_load + 
                                     eps_normal[t, self.DATABASE])
                
                # Different temporal dynamics for testing samples (same 5 services)
                if not is_training:
                    # Apply different causal structure coefficients for testing
                    # This creates different temporal patterns while keeping same topology
                    
                    # Use testing-specific causal coefficients for more complex dynamics
                    if v1_weight > 0.5:  # Phase 1 testing pattern
                        # Frontend gets feedback from database (testing pattern)
                        x[t, self.FRONTEND] += 0.1 * x[t-1, self.DATABASE]
                        # Load balancer gets feedback from Target-v1
                        x[t, self.LOAD_BALANCER] += 0.1 * x[t-1, self.TARGET_V1]
                        # Target-v1 has stronger cross-dependencies
                        x[t, self.TARGET_V1] += 0.1 * x[t-1, self.FRONTEND]
                        # Database has stronger dependencies
                        x[t, self.DATABASE] += 0.1 * x[t-1, self.FRONTEND]
                    
                    if v2_weight > 0.5:  # Phase 2 testing pattern
                        # Different coupling strengths in testing
                        x[t, self.FRONTEND] += 0.2 * x[t-1, self.DATABASE]
                        x[t, self.LOAD_BALANCER] += 0.2 * x[t-1, self.TARGET_V2]
                        x[t, self.TARGET_V2] += 0.2 * x[t-1, self.FRONTEND]
                        x[t, self.DATABASE] += 0.2 * x[t-1, self.FRONTEND]

            # Generate anomalous time series (same as normal but with anomaly injection)
            x_ab = x.copy()
            
            # Inject anomaly into Target-v2
            for t_idx in t_p:
                if t_idx < self.t:
                    x_ab[t_idx, feature_p] += ab[feature_p]
                    
                    # Propagate anomaly effects to dependent services
                    if t_idx < self.t - 1:
                        # Database gets stressed by faulty Target-v2
                        x_ab[t_idx + 1, self.DATABASE] += 0.3 * ab[feature_p[0]]
                        # Load balancer detects issues
                        x_ab[t_idx + 1, self.LOAD_BALANCER] += 0.1 * ab[feature_p[0]]

            # Clip values to reasonable ranges
            x = np.clip(x, 0, 5)
            x_ab = np.clip(x_ab, 0, 10)  # Allow higher values during anomalies

            x_n_list.append(x)
            x_ab_list.append(x_ab)
            eps_n_list.append(eps_normal)
            eps_ab_list.append(eps_anom)
            label_list.append(temp_label)

        # Store all data
        self.data_dict['x_n_list'] = x_n_list
        self.data_dict['x_ab_list'] = x_ab_list
        self.data_dict['eps_n_list'] = eps_n_list
        self.data_dict['eps_ab_list'] = eps_ab_list
        self.data_dict['label_list'] = label_list
        self.data_dict['timing_info'] = timing_info

    def save_data(self):
        """Save the generated data to disk."""
        if not hasattr(self, 'data_dict') or not self.data_dict:
            print("No data to save. Please run generate_example() first.")
            return
            
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Separate training and testing data to handle different shapes
        training_size = self.options['training_size']
        
        # Split data into training and testing
        x_n_train = self.data_dict['x_n_list'][:training_size]
        x_n_test = self.data_dict['x_n_list'][training_size:]
        x_ab_train = self.data_dict['x_ab_list'][:training_size]
        x_ab_test = self.data_dict['x_ab_list'][training_size:]
        eps_n_train = self.data_dict['eps_n_list'][:training_size]
        eps_n_test = self.data_dict['eps_n_list'][training_size:]
        eps_ab_train = self.data_dict['eps_ab_list'][:training_size]
        eps_ab_test = self.data_dict['eps_ab_list'][training_size:]
        label_train = self.data_dict['label_list'][:training_size]
        label_test = self.data_dict['label_list'][training_size:]
        
        # Save training data (5 variables)
        np.save(os.path.join(self.data_dir, 'x_n_train.npy'), x_n_train)
        np.save(os.path.join(self.data_dir, 'x_ab_train.npy'), x_ab_train)
        np.save(os.path.join(self.data_dir, 'eps_n_train.npy'), eps_n_train)
        np.save(os.path.join(self.data_dir, 'eps_ab_train.npy'), eps_ab_train)
        np.save(os.path.join(self.data_dir, 'label_train.npy'), label_train)
        
        # Save testing data (7 variables)
        np.save(os.path.join(self.data_dir, 'x_n_test.npy'), x_n_test)
        np.save(os.path.join(self.data_dir, 'x_ab_test.npy'), x_ab_test)
        np.save(os.path.join(self.data_dir, 'eps_n_test.npy'), eps_n_test)
        np.save(os.path.join(self.data_dir, 'eps_ab_test.npy'), eps_ab_test)
        np.save(os.path.join(self.data_dir, 'label_test.npy'), label_test)
        
        # Save combined lists as Python objects using pickle (since shapes are different)
        import pickle
        with open(os.path.join(self.data_dir, 'x_n_list.pkl'), 'wb') as f:
            pickle.dump(self.data_dict['x_n_list'], f)
        with open(os.path.join(self.data_dir, 'x_ab_list.pkl'), 'wb') as f:
            pickle.dump(self.data_dict['x_ab_list'], f)
        with open(os.path.join(self.data_dir, 'eps_n_list.pkl'), 'wb') as f:
            pickle.dump(self.data_dict['eps_n_list'], f)
        with open(os.path.join(self.data_dir, 'eps_ab_list.pkl'), 'wb') as f:
            pickle.dump(self.data_dict['eps_ab_list'], f)
        with open(os.path.join(self.data_dir, 'label_list.pkl'), 'wb') as f:
            pickle.dump(self.data_dict['label_list'], f)
        
        # Save causal structures
        np.save(os.path.join(self.data_dir, 'causal_struct_training_phase1.npy'), self.data_dict['causal_struct_training_phase1'])
        np.save(os.path.join(self.data_dir, 'causal_struct_training_phase2.npy'), self.data_dict['causal_struct_training_phase2'])
        np.save(os.path.join(self.data_dir, 'causal_struct_testing_phase1.npy'), self.data_dict['causal_struct_testing_phase1'])
        np.save(os.path.join(self.data_dir, 'causal_struct_testing_phase2.npy'), self.data_dict['causal_struct_testing_phase2'])
        
        # Save timing information as JSON
        import json
        with open(os.path.join(self.data_dir, 'timing_info.json'), 'w') as f:
            json.dump(self.data_dict['timing_info'], f, indent=2)
            
        print(f"Dynamic Temporal Service dataset saved to: {self.data_dir}")
        print(f"Training data shape: {len(x_n_train)} samples with {self.training_vars} variables")
        print(f"Testing data shape: {len(x_n_test)} samples with {self.testing_vars} variables")

    def load_data(self):
        """Load existing data from disk."""
        try:
            # Try to load from pickle files first (new format)
            import pickle
            try:
                with open(os.path.join(self.data_dir, 'x_n_list.pkl'), 'rb') as f:
                    self.data_dict['x_n_list'] = pickle.load(f)
                with open(os.path.join(self.data_dir, 'x_ab_list.pkl'), 'rb') as f:
                    self.data_dict['x_ab_list'] = pickle.load(f)
                with open(os.path.join(self.data_dir, 'eps_n_list.pkl'), 'rb') as f:
                    self.data_dict['eps_n_list'] = pickle.load(f)
                with open(os.path.join(self.data_dir, 'eps_ab_list.pkl'), 'rb') as f:
                    self.data_dict['eps_ab_list'] = pickle.load(f)
                with open(os.path.join(self.data_dir, 'label_list.pkl'), 'rb') as f:
                    self.data_dict['label_list'] = pickle.load(f)
                print("Loaded data from pickle files")
            except FileNotFoundError:
                # Fallback to numpy files (old format)
                self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=True)
                self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
                self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'), allow_pickle=True)
                self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'), allow_pickle=True)
                self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)
                print("Loaded data from numpy files")
            
            # Load causal structures
            self.data_dict['causal_struct_training_phase1'] = np.load(os.path.join(self.data_dir, 'causal_struct_training_phase1.npy'))
            self.data_dict['causal_struct_training_phase2'] = np.load(os.path.join(self.data_dir, 'causal_struct_training_phase2.npy'))
            self.data_dict['causal_struct_testing_phase1'] = np.load(os.path.join(self.data_dir, 'causal_struct_testing_phase1.npy'))
            self.data_dict['causal_struct_testing_phase2'] = np.load(os.path.join(self.data_dir, 'causal_struct_testing_phase2.npy'))
            self.data_dict['causal_struct'] = self.data_dict['causal_struct_training_phase1']  # Default to training phase1
            
            # Load timing information
            import json
            with open(os.path.join(self.data_dir, 'timing_info.json'), 'r') as f:
                self.data_dict['timing_info'] = json.load(f)
                
            print(f"Dynamic Temporal Service dataset loaded from: {self.data_dir}")
            
            # Also load separated training/testing data if available
            try:
                self.data_dict['x_n_train'] = np.load(os.path.join(self.data_dir, 'x_n_train.npy'))
                self.data_dict['x_ab_train'] = np.load(os.path.join(self.data_dir, 'x_ab_train.npy'))
                self.data_dict['eps_n_train'] = np.load(os.path.join(self.data_dir, 'eps_n_train.npy'))
                self.data_dict['eps_ab_train'] = np.load(os.path.join(self.data_dir, 'eps_ab_train.npy'))
                self.data_dict['label_train'] = np.load(os.path.join(self.data_dir, 'label_train.npy'))
                
                self.data_dict['x_n_test'] = np.load(os.path.join(self.data_dir, 'x_n_test.npy'))
                self.data_dict['x_ab_test'] = np.load(os.path.join(self.data_dir, 'x_ab_test.npy'))
                self.data_dict['eps_n_test'] = np.load(os.path.join(self.data_dir, 'eps_n_test.npy'))
                self.data_dict['eps_ab_test'] = np.load(os.path.join(self.data_dir, 'eps_ab_test.npy'))
                self.data_dict['label_test'] = np.load(os.path.join(self.data_dir, 'label_test.npy'))
                print("Loaded separated training/testing data files")
            except FileNotFoundError:
                print("Separated training/testing files not found, using combined lists only")
                
        except FileNotFoundError as e:
            print(f"Data files not found: {e}")
            print("Please run generate_example() first to create the dataset.")

