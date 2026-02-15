import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random


class SMD:
    def __init__(self, options):
        """
        Initializes the SMD (Server Machine Dataset) class.
        
        Args:
            options (dict): Configuration options for the dataset.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options.get('num_vars', 38)  # Default number of features
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options.get('shuffle', True)
        self.selected_machines = options.get('selected_machines', None)  # Specific machines to use
        self.machine_groups = options.get('machine_groups', [1, 2, 3])  # Which machine groups to include
        
    def generate_example(self):
        """
        Loads and processes the SMD dataset from text files.
        """
        # Get available machine files
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        label_dir = os.path.join(self.data_dir, 'test_label')
        
        available_machines = []
        for file in os.listdir(train_dir):
            if file.endswith('.txt'):
                machine_id = file.replace('.txt', '')
                # Check if machine belongs to selected groups
                machine_group = int(machine_id.split('-')[1])
                if machine_group in self.machine_groups:
                    available_machines.append(machine_id)
        
        # Use selected machines or all available machines
        if self.selected_machines:
            machines_to_use = [m for m in self.selected_machines if m in available_machines]
        else:
            # Limit to a few machines from each group for manageable size
            machines_to_use = []
            for group in self.machine_groups:
                group_machines = [m for m in available_machines if m.startswith(f'machine-{group}-')]
                machines_to_use.extend(group_machines[:1])  # Take first 1 from each group
                
        print(f"Using SMD machines: {machines_to_use}")
        
        x_n_list = []
        x_ab_list = []
        label_list = []
        
        for machine_id in machines_to_use:
            # Load training (normal) data
            train_data = pd.read_csv(os.path.join(train_dir, f'{machine_id}.txt'), header=None).values
            test_data = pd.read_csv(os.path.join(test_dir, f'{machine_id}.txt'), header=None).values
            labels = pd.read_csv(os.path.join(label_dir, f'{machine_id}.txt'), header=None).values.flatten()
            
            # Limit features if specified
            if train_data.shape[1] > self.num_vars:
                train_data = train_data[:, :self.num_vars]
                test_data = test_data[:, :self.num_vars]
            
            # Create normal sequences from training data (reduced number for faster training)
            # Add extra length for sliding window operations (window_size + 1 needed)
            # Use reasonable sequence length since we generate multiple sequences (no temporal splitting)
            sequence_length = self.window_size + 30  # Sufficient buffer for sliding window operations
            
            # Limit normal sequences per machine
            max_sequences_per_machine = 50
            step_size = max(1, (len(train_data) - sequence_length) // max_sequences_per_machine)
            count = 0
            for i in range(0, len(train_data) - sequence_length + 1, step_size):
                if count >= max_sequences_per_machine:
                    break
                if i + sequence_length <= len(train_data):
                    x_n_list.append(train_data[i:i + sequence_length])
                    count += 1
            
            # Find anomaly regions in test data
            anomaly_indices = np.where(labels == 1)[0]
            
            if len(anomaly_indices) > 0:
                # Group consecutive anomaly indices
                anomaly_groups = []
                current_group = [anomaly_indices[0]]
                
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] - anomaly_indices[i-1] == 1:
                        current_group.append(anomaly_indices[i])
                    else:
                        anomaly_groups.append(current_group)
                        current_group = [anomaly_indices[i]]
                anomaly_groups.append(current_group)
                
                # Create sequences containing anomalies
                for group in anomaly_groups:
                    if len(group) > 0:
                        start_idx = group[0]
                        end_idx = group[-1]
                        
                        # Create a sequence that includes context before and after anomaly
                        context_before = sequence_length // 3
                        context_after = sequence_length // 3
                        
                        seq_start = max(0, start_idx - context_before)
                        seq_end = min(len(test_data), seq_start + sequence_length)
                        
                        if seq_end - seq_start >= sequence_length:
                            sequence = test_data[seq_start:seq_end]
                            x_ab_list.append(sequence)
                            
                            # Create labels for this sequence
                            seq_labels = labels[seq_start:seq_end]
                            # Expand labels to match feature dimensions
                            expanded_labels = np.tile(seq_labels.reshape(-1, 1), (1, sequence.shape[1]))
                            label_list.append(expanded_labels)
        
        # Normalize data
        scaler = MinMaxScaler()
        
        # Fit scaler on normal data
        if x_n_list:
            all_normal_data = np.concatenate(x_n_list, axis=0)
            scaler.fit(all_normal_data)
            
            # Transform data
            x_n_list = [scaler.transform(seq) for seq in x_n_list]
            x_ab_list = [scaler.transform(seq) for seq in x_ab_list]
        
        # Convert to numpy arrays
        self.data_dict['x_n_list'] = np.array(x_n_list) if x_n_list else np.array([])
        self.data_dict['x_ab_list'] = np.array(x_ab_list) if x_ab_list else np.array([])
        self.data_dict['label_list'] = np.array(label_list) if label_list else np.array([])
        
        # Shuffle if requested
        if self.shuffle and len(self.data_dict['x_n_list']) > 0:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
            
        print(f"SMD dataset loaded: {len(self.data_dict['x_n_list'])} normal sequences, "
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
