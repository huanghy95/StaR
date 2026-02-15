import argparse
import os

def create_arg_parser():
    """
    Creates and returns the argument parser for the AIOps Node Disk Fill dataset.

    Returns:
        argparse.ArgumentParser: The argument parser for the AIOps Node Disk Fill dataset.
    """
    parser = argparse.ArgumentParser(description='AIOps Node Disk Fill Dataset')

    # Dataset arguments
    parser.add_argument('--num_vars', type=int, default=11, help='Number of node variables (default: 11)')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for sequences (default: 20)')
    parser.add_argument('--shuffle', type=int, default=1, help='Shuffle data (default: 1)')
    parser.add_argument('--preprocessing_data', type=int, default=1, help='Flag for preprocessing data (default: 1)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets', 'aiops_node_disk_fill'), 
                       help='Data directory (default: ./datasets/aiops_node_disk_fill)')
    parser.add_argument('--causal_quantile', type=float, default=0.50, help='Causal quantile (default: 0.50)')
    
    # AIOps specific arguments
    parser.add_argument('--fault_type', type=str, default='node_disk_fill', help='Type of fault to analyze (default: node_disk_fill)')
    parser.add_argument('--time_series_length', type=int, default=200, help='Length of time series to generate (default: 200)')
    parser.add_argument('--anomaly_ratio', type=float, default=0.8, help='Ratio of anomalous samples (default: 0.8)')
    parser.add_argument('--training_size', type=int, default=10, help='Number of training samples (default: 10)')
    parser.add_argument('--testing_size', type=int, default=20, help='Number of testing samples (default: 20)')

    # Meta arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default='aiops', help='Dataset name (default: aiops)')

    # AERCA arguments
    parser.add_argument('--stride', type=int, default=1, help='Stride (default: 1)')
    parser.add_argument('--encoder_alpha', type=float, default=0.5, help='Encoder alpha (default: 0.5)')
    parser.add_argument('--decoder_alpha', type=float, default=0.5, help='Decoder alpha (default: 0.5)')
    parser.add_argument('--encoder_gamma', type=float, default=0.5, help='Encoder gamma (default: 0.5)')
    parser.add_argument('--decoder_gamma', type=float, default=0.5, help='Decoder gamma (default: 0.5)')
    parser.add_argument('--encoder_lambda', type=float, default=0.5, help='Encoder lambda (default: 0.5)')
    parser.add_argument('--decoder_lambda', type=float, default=0.5, help='Decoder lambda (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs (default: 2000)')
    parser.add_argument('--hidden_layer_size', type=int, default=32, help='Hidden layer size (default: 32)')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--recon_threshold', type=float, default=0.95, help='Reconstruction threshold (default: 0.95)')
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.95,
                       help='Root cause threshold for encoder (default: 0.95)')
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.95,
                       help='Root cause threshold for decoder (default: 0.95)')
    parser.add_argument('--training_aerca', type=int, default=1, help='Enable AERCA training (default: 1)')
    parser.add_argument('--initial_z_score', type=float, default=3.0, help='Initial z-score for POT (default: 3.0)')
    parser.add_argument('--risk', type=float, default=1e-2, help='Risk parameter for POT (default: 1e-2)')
    parser.add_argument('--initial_level', type=float, default=0.98, help='Initial level for POT (default: 0.98)')
    parser.add_argument('--num_candidates', type=int, default=100, help='Number of candidates (default: 100)')

    return parser

