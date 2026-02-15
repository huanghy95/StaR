import argparse
import os

def create_arg_parser():
    """
    Creates and returns the argument parser for the Dynamic Temporal Service dataset.
    
    This dataset simulates a microservice architecture with dynamic dependency changes:
    - Variable timing for phase transitions (not fixed like original)
    - Gradual sigmoid transitions instead of abrupt changes
    - Memory-dependent service states (TGN advantage)
    - Non-linear service interactions under stress (KGC advantage)
    - Different causal patterns and temporal dynamics between training vs testing
    - Same 5 services used throughout, but with different interaction patterns

    Returns:
        argparse.ArgumentParser: The argument parser for the Dynamic Temporal Service dataset.
    """
    parser = argparse.ArgumentParser(description='Dynamic Temporal Service Dataset for StaR')

    # Dataset arguments
    parser.add_argument('--T', type=int, default=5000, help='Length of the time series (default: 5000)')
    parser.add_argument('--training_size', type=int, default=10, help='Size of the training set (default: 10)')
    parser.add_argument('--testing_size', type=int, default=100, help='Size of the testing set (default: 100)')
    parser.add_argument('--num_vars', type=int, default=5, help='Number of services for both training and testing (default: 5)')
    parser.add_argument('--preprocessing_data', type=int, default=1, help='Flag for preprocessing data (default: 1)')
    parser.add_argument('--adlength', type=int, default=10, help='Anomaly duration (default: 10)')
    parser.add_argument('--adtype', type=str, default='causal', help='Anomaly type: non_causal or causal (default: causal)')
    parser.add_argument('--mul', type=int, default=6, help='Anomaly magnitude multiplier (default: 6)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets', 'temporal_service_dynamic'), 
                       help='Data directory (default: ./datasets/temporal_service_dynamic)')
    parser.add_argument('--causal_quantile', type=float, default=0.75, help='Causal quantile threshold (default: 0.75)')
    parser.add_argument('--noise_scale', type=float, default=0.3, help='Noise scale (default: 0.3)')
    parser.add_argument('--dependent_features', type=int, default=1, help='Flag for dependent features (default: 1)')

    # Meta arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default='temporal_service_dynamic', help='Dataset name')

    # AERCA arguments
    parser.add_argument('--window_size', type=int, default=8, help='Window size for temporal analysis (default: 8)')
    parser.add_argument('--stride', type=int, default=1, help='Stride (default: 1)')
    parser.add_argument('--encoder_alpha', type=float, default=0.5, help='Encoder alpha (default: 0.5)')
    parser.add_argument('--decoder_alpha', type=float, default=0.5, help='Decoder alpha (default: 0.5)')
    parser.add_argument('--encoder_gamma', type=float, default=0.6, help='Encoder gamma (default: 0.6)')
    parser.add_argument('--decoder_gamma', type=float, default=0.6, help='Decoder gamma (default: 0.6)')
    parser.add_argument('--encoder_lambda', type=float, default=0.4, help='Encoder lambda (default: 0.4)')
    parser.add_argument('--decoder_lambda', type=float, default=0.4, help='Decoder lambda (default: 0.4)')
    parser.add_argument('--beta', type=float, default=0.3, help='Beta (default: 0.3)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs (default: 2000)')
    parser.add_argument('--hidden_layer_size', type=int, default=64, help='Hidden layer size (default: 64)')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='Number of hidden layers (default: 4)')
    parser.add_argument('--recon_threshold', type=float, default=0.90, help='Reconstruction threshold (default: 0.90)')
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.95, help='Root cause threshold for encoder (default: 0.95)')
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.95, help='Root cause threshold for decoder (default: 0.95)')
    parser.add_argument('--training_aerca', type=int, default=1, help='Flag for training AERCA (default: 1)')
    parser.add_argument('--initial_z_score', type=float, default=2.5, help='Initial Z-score (default: 2.5)')
    parser.add_argument('--risk', type=float, default=1e-2, help='Risk (default: 1e-2)')
    parser.add_argument('--initial_level', type=float, default=0.95, help='Initial level (default: 0.95)')
    parser.add_argument('--num_candidates', type=int, default=100, help='Number of candidates (default: 100)')

    return parser

def get_default_options():
    """
    Returns default options for the dynamic temporal service dataset as a dictionary.
    """
    return {
        'T': 5000,
        'training_size': 10,
        'testing_size': 100,
        'num_vars': 5,
        'preprocessing_data': 1,
        'adlength': 10,
        'adtype': 'causal',
        'mul': 6,
        'data_dir': os.path.join(os.getcwd(), 'datasets', 'temporal_service_dynamic'),
        'causal_quantile': 0.75,
        'noise_scale': 0.3,
        'dependent_features': 1,
        'seed': 42,
        'device': 'cuda',
        'dataset_name': 'temporal_service_dynamic',
        'window_size': 8,
        'stride': 1,
        'encoder_alpha': 0.5,
        'decoder_alpha': 0.5,
        'encoder_gamma': 0.6,
        'decoder_gamma': 0.6,
        'encoder_lambda': 0.4,
        'decoder_lambda': 0.4,
        'beta': 0.3,
        'lr': 0.0001,
        'epochs': 2000,
        'hidden_layer_size': 64,
        'num_hidden_layers': 4,
        'recon_threshold': 0.90,
        'root_cause_threshold_encoder': 0.95,
        'root_cause_threshold_decoder': 0.95,
        'training_aerca': 1,
        'initial_z_score': 2.5,
        'risk': 1e-2,
        'initial_level': 0.95,
        'num_candidates': 100
    }

