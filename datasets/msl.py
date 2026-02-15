import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import ast


class MSL:
    def __init__(self, options):
        """
        Initializes the MSL (Mars Science Laboratory) dataset class.
        
        Args:
            options (dict): Configuration options for the dataset.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options.get('num_vars', 25)  # Default number of features
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options.get('shuffle', True)
        self.selected_channels = options.get('selected_channels', None)  # Specific channels to use
        
    def generate_example(self):
        """
        Loads and processes the MSL dataset from numpy files and labels.
        """
        # Load labeled anomalies information
        labels_df = pd.read_csv(os.path.join(self.data_dir, 'labeled_anomalies.csv'))
        
        # Filter for MSL spacecraft only
        msl_labels = labels_df[labels_df['spacecraft'] == 'MSL']
        
        # Get available channels
        train_dir = os.path.join(self.data_dir, 'data', 'data', 'train')
        test_dir = os.path.join(self.data_dir, 'data', 'data', 'test')
        
        available_channels = []
        for file in os.listdir(train_dir):
            if file.endswith('.npy'):
                channel_id = file.replace('.npy', '')
                if channel_id in msl_labels['chan_id'].values:
                    available_channels.append(channel_id)
        
        # Use selected channels or all available MSL channels
        if self.selected_channels:
            channels_to_use = [ch for ch in self.selected_channels if ch in available_channels]
        else:
            channels_to_use = available_channels[:5]  # Limit to first 5 channels for manageable size
            
        print(f"Using MSL channels: {channels_to_use}")
        
        x_n_list = []
        x_ab_list = []
        label_list = []
        
        for channel_id in channels_to_use:
            # Load training (normal) data
            train_data = np.load(os.path.join(train_dir, f'{channel_id}.npy'))
            test_data = np.load(os.path.join(test_dir, f'{channel_id}.npy'))
            
            # Limit features if specified
            if train_data.shape[1] > self.num_vars:
                train_data = train_data[:, :self.num_vars]
                test_data = test_data[:, :self.num_vars]
            
            # Create normal sequences from training data (reduced number for faster training)
            # Add extra length for sliding window operations (window_size + 1 needed)
            # Use reasonable sequence length since we generate multiple sequences (no temporal splitting)
            sequence_length = self.window_size + 30  # Sufficient buffer for sliding window operations
            
            # Limit normal sequences per channel
            max_sequences_per_channel = 30
            step_size = max(1, (len(train_data) - sequence_length) // max_sequences_per_channel)
            count = 0
            for i in range(0, len(train_data) - sequence_length + 1, step_size):
                if count >= max_sequences_per_channel:
                    break
                if i + sequence_length <= len(train_data):
                    x_n_list.append(train_data[i:i + sequence_length])
                    count += 1
            
            # Get anomaly information for this channel
            channel_info = msl_labels[msl_labels['chan_id'] == channel_id]
            if not channel_info.empty:
                anomaly_sequences = ast.literal_eval(channel_info.iloc[0]['anomaly_sequences'])
                
                # Create anomaly sequences
                for anomaly_range in anomaly_sequences:
                    start_idx, end_idx = anomaly_range
                    
                    # Ensure we have enough data before and after anomaly
                    context_before = self.window_size // 2
                    context_after = self.window_size // 2
                    
                    seq_start = max(0, start_idx - context_before)
                    seq_end = min(len(test_data), end_idx + context_after)
                    
                    if seq_end - seq_start >= sequence_length:
                        # Take a window that includes the anomaly
                        sequence = test_data[seq_start:seq_start + sequence_length]
                        x_ab_list.append(sequence)
                        
                        # Create labels
                        labels = np.zeros((sequence_length, sequence.shape[1]))
                        
                        # Mark anomaly region
                        anomaly_start_in_seq = max(0, start_idx - seq_start)
                        anomaly_end_in_seq = min(sequence_length, end_idx - seq_start)
                        
                        if anomaly_start_in_seq < sequence_length and anomaly_end_in_seq > 0:
                            labels[anomaly_start_in_seq:anomaly_end_in_seq] = 1
                            
                        label_list.append(labels)
        
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
            
        print(f"MSL dataset loaded: {len(self.data_dict['x_n_list'])} normal sequences, "
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
