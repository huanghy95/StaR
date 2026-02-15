import argparse
import os

def create_arg_parser():
    """
    Creates and returns the argument parser for the Random Connection Service dataset.
    
    This dataset simulates a 5-service architecture with correct dependency flow:
    - Service dependencies: service 2 <- service 1 <- service 3/4/5
    - Edge services (3, 4, 5) can randomly connect TO service 1
    - Service 1 is always connected to service 2 (stable backbone)
    - Root cause is randomly assigned to one or more of services [3, 4, 5]
    - Problems occur when root cause services connect to service 1
    
    This design is particularly favorable to TGN models due to:
    - Dynamic graph structure that changes at each time step
    - Memory effects from past connection patterns
    - Complex temporal dependencies requiring connection history tracking
    - Realistic dependency flow from edge services to core infrastructure

    Returns:
        argparse.ArgumentParser: The argument parser for the Random Connection Service dataset.
    """
    parser = argparse.ArgumentParser(description='Random Connection Service Dataset for StaR')

    # Dataset arguments
    parser.add_argument('--T', type=int, default=4000, help='Length of the time series (default: 4000)')
    parser.add_argument('--training_size', type=int, default=15, help='Size of the training set (default: 15)')
    parser.add_argument('--testing_size', type=int, default=100, help='Size of the testing set (default: 100)')
    parser.add_argument('--num_vars', type=int, default=5, help='Number of services (default: 5)')
    parser.add_argument('--preprocessing_data', type=int, default=1, help='Flag for preprocessing data (default: 1)')
    parser.add_argument('--adlength', type=int, default=8, help='Anomaly duration (default: 8)')
    parser.add_argument('--adtype', type=str, default='causal', help='Anomaly type: non_causal or causal (default: causal)')
    parser.add_argument('--mul', type=int, default=4, help='Anomaly magnitude multiplier (default: 4)')
    parser.add_argument('--root_cause_strategy', type=str, default='single_variable', 
                       help='Root cause assignment strategy (default: single_variable)')
    parser.add_argument('--fixed_root_cause', type=int, default=None, choices=[3, 4, 5],
                       help='Fixed root cause service for all samples (3, 4, or 5)')
    parser.add_argument('--training_root_cause', type=int, default=None, choices=[3, 4, 5],
                       help='Fixed root cause service for training samples (3, 4, or 5)')
    parser.add_argument('--testing_root_cause', type=int, default=None, choices=[3, 4, 5],
                       help='Fixed root cause service for testing samples (3, 4, or 5)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets', 'random_connection'), 
                       help='Data directory (default: ./datasets/random_connection)')
    parser.add_argument('--causal_quantile', type=float, default=0.70, help='Causal quantile threshold (default: 0.70)')
    parser.add_argument('--noise_scale', type=float, default=0.25, help='Noise scale (default: 0.25)')
    parser.add_argument('--dependent_features', type=int, default=1, help='Flag for dependent features (default: 1)')
    parser.add_argument('--base_connection_prob', type=float, default=0.65,
                       help='Base probability that edge services connect to service 1 (default: 0.65)')

    # Meta arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default='random_connection', help='Dataset name')

    # AERCA arguments
    parser.add_argument('--window_size', type=int, default=10, help='Window size for temporal analysis (default: 10)')
    parser.add_argument('--stride', type=int, default=1, help='Stride (default: 1)')
    parser.add_argument('--encoder_alpha', type=float, default=0.4, help='Encoder alpha (default: 0.4)')
    parser.add_argument('--decoder_alpha', type=float, default=0.4, help='Decoder alpha (default: 0.4)')
    parser.add_argument('--encoder_gamma', type=float, default=0.7, help='Encoder gamma (default: 0.7)')
    parser.add_argument('--decoder_gamma', type=float, default=0.7, help='Decoder gamma (default: 0.7)')
    parser.add_argument('--encoder_lambda', type=float, default=0.3, help='Encoder lambda (default: 0.3)')
    parser.add_argument('--decoder_lambda', type=float, default=0.3, help='Decoder lambda (default: 0.3)')
    parser.add_argument('--beta', type=float, default=0.2, help='Beta (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs (default: 1500)')
    parser.add_argument('--hidden_layer_size', type=int, default=48, help='Hidden layer size (default: 48)')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--recon_threshold', type=float, default=0.85, help='Reconstruction threshold (default: 0.85)')
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.90, help='Root cause threshold for encoder (default: 0.90)')
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.90, help='Root cause threshold for decoder (default: 0.90)')
    parser.add_argument('--training_aerca', type=int, default=1, help='Flag for training AERCA (default: 1)')
    parser.add_argument('--initial_z_score', type=float, default=2.0, help='Initial Z-score (default: 2.0)')
    parser.add_argument('--risk', type=float, default=1e-2, help='Risk (default: 1e-2)')
    parser.add_argument('--initial_level', type=float, default=0.90, help='Initial level (default: 0.90)')
    parser.add_argument('--num_candidates', type=int, default=100, help='Number of candidates (default: 100)')

    return parser

def get_default_options():
    """
    Returns default options for the random connection service dataset as a dictionary.
    """
    return {
        'T': 4000,
        'training_size': 15,
        'testing_size': 100,
        'num_vars': 5,
        'preprocessing_data': 1,
        'adlength': 8,
        'adtype': 'causal',
        'mul': 4,
        'data_dir': os.path.join(os.getcwd(), 'datasets', 'random_connection'),
        'causal_quantile': 0.70,
        'noise_scale': 0.25,
        'dependent_features': 1,
        'base_connection_prob': 0.65,
        'seed': 42,
        'device': 'cuda',
        'dataset_name': 'random_connection',
        'window_size': 10,
        'stride': 1,
        'encoder_alpha': 0.4,
        'decoder_alpha': 0.4,
        'encoder_gamma': 0.7,
        'decoder_gamma': 0.7,
        'encoder_lambda': 0.3,
        'decoder_lambda': 0.3,
        'beta': 0.2,
        'lr': 0.0001,
        'epochs': 1500,
        'hidden_layer_size': 48,
        'num_hidden_layers': 2,
        'recon_threshold': 0.85,
        'root_cause_threshold_encoder': 0.90,
        'root_cause_threshold_decoder': 0.90,
        'training_aerca': 1,
        'initial_z_score': 2.0,
        'risk': 1e-2,
        'initial_level': 0.90,
        'num_candidates': 100
    }

