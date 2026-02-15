import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pyreadr
import os
import random


class TEP:
    def __init__(self, options):
        """
        Initializes the TEP (Tennessee Eastman Process) dataset class.
        
        Args:
            options (dict): Configuration options for the dataset.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options.get('num_vars', 52)  # 52 process variables (excluding faultNumber, simulationRun, sample)
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options.get('shuffle', True)
        self.fault_types = options.get('fault_types', [1, 2, 3, 4, 5])  # Which fault types to include
        self.samples_per_fault = options.get('samples_per_fault', 100)  # Number of samples per fault type
        
    def generate_example(self):
        """
        Loads and processes the TEP dataset from RData files.
        """
        # Load RData files
        tep_normal_train = pyreadr.read_r(os.path.join(self.data_dir, 'TEP_FaultFree_Training.RData'))
        tep_normal_test = pyreadr.read_r(os.path.join(self.data_dir, 'TEP_FaultFree_Testing.RData'))
        tep_faulty_train = pyreadr.read_r(os.path.join(self.data_dir, 'TEP_Faulty_Training.RData'))
        tep_faulty_test = pyreadr.read_r(os.path.join(self.data_dir, 'TEP_Faulty_Testing.RData'))
        
        # Extract dataframes
        normal_train_df = list(tep_normal_train.values())[0]
        normal_test_df = list(tep_normal_test.values())[0]
        faulty_train_df = list(tep_faulty_train.values())[0]
        faulty_test_df = list(tep_faulty_test.values())[0]
        
        # Select process variables (exclude faultNumber, simulationRun, sample)
        process_vars = [col for col in normal_train_df.columns if col.startswith(('xmeas_', 'xmv_'))]
        process_vars = process_vars[:self.num_vars]  # Limit to specified number of variables
        
        # Process normal data for training
        normal_data = normal_train_df[process_vars].values
        
        # Create windowed sequences from normal data
        x_n_list = []
        # Add extra length for sliding window operations (window_size + 1 needed)
        # Use reasonable sequence length since we generate multiple sequences (no temporal splitting)
        sequence_length = self.window_size + 30  # Sufficient buffer for sliding window operations
        
        # Generate normal sequences (reduced number for faster training)
        max_normal_sequences = 200  # Limit normal sequences
        step_size = max(1, (len(normal_data) - sequence_length) // max_normal_sequences)
        count = 0
        for i in range(0, len(normal_data) - sequence_length + 1, step_size):
            if count >= max_normal_sequences:
                break
            if i + sequence_length <= len(normal_data):
                x_n_list.append(normal_data[i:i + sequence_length])
                count += 1
        
        # Process faulty data for testing
        x_ab_list = []
        label_list = []
        
        for fault_type in self.fault_types:
            # Get faulty data for this fault type
            fault_data = faulty_test_df[faulty_test_df['faultNumber'] == fault_type]
            
            # Group by simulation run
            simulation_runs = fault_data['simulationRun'].unique()
            
            count = 0
            for sim_run in simulation_runs:
                if count >= self.samples_per_fault:
                    break
                    
                sim_data = fault_data[fault_data['simulationRun'] == sim_run][process_vars].values
                
                if len(sim_data) >= sequence_length:
                    # Take the sequence where fault occurs (usually after some normal operation)
                    # TEP faults typically start after 160 samples (8 hours)
                    fault_start = min(160, len(sim_data) - sequence_length)
                    
                    # Create sequence with fault
                    sequence = sim_data[fault_start:fault_start + sequence_length]
                    x_ab_list.append(sequence)
                    
                    # Create labels (1 for anomaly, 0 for normal)
                    # Assume fault starts at 20% of the sequence
                    labels = np.zeros((sequence_length, self.num_vars))
                    fault_start_in_seq = int(0.2 * sequence_length)
                    labels[fault_start_in_seq:] = 1  # Mark as anomaly from fault start
                    label_list.append(labels)
                    
                    count += 1
        
        # Normalize data
        scaler = MinMaxScaler()
        
        # Fit scaler on normal data
        all_normal_data = np.concatenate(x_n_list, axis=0)
        scaler.fit(all_normal_data)
        
        # Transform data
        x_n_list = [scaler.transform(seq) for seq in x_n_list]
        x_ab_list = [scaler.transform(seq) for seq in x_ab_list]
        
        # Convert to numpy arrays
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['label_list'] = np.array(label_list)
        
        # Shuffle if requested
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
            
        print(f"TEP dataset loaded: {len(self.data_dict['x_n_list'])} normal sequences, "
              f"{len(self.data_dict['x_ab_list'])} anomaly sequences")
        
    def save_data(self):
        """Save processed data to numpy files."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])
        
    def load_data(self):
        """Load processed data from numpy files."""
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=False)
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=False)
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=False)
