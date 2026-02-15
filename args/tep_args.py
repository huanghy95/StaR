import argparse
import os

def create_arg_parser():
    """
    Creates and returns the argument parser for the TEP dataset.

    Returns:
        argparse.ArgumentParser: The argument parser for the TEP dataset.
    """
    parser = argparse.ArgumentParser(description='Tennessee Eastman Process (TEP) Dataset')

    # Dataset arguments
    parser.add_argument('--num_vars', type=int, default=25, help='Number of process variables (default: 25)')
    parser.add_argument('--window_size', type=int, default=50, help='Window size for sequences (default: 50)')
    parser.add_argument('--fault_types', type=int, nargs='+', default=[1, 2, 3], 
                       help='Fault types to include (default: [1, 2, 3])')
    parser.add_argument('--samples_per_fault', type=int, default=20, 
                       help='Number of samples per fault type (default: 20)')
    parser.add_argument('--shuffle', type=int, default=1, help='Shuffle data (default: 1)')
    parser.add_argument('--preprocessing_data', type=int, default=1, help='Flag for preprocessing data (default: 1)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets', 'tep'), 
                       help='Data directory (default: ./datasets/tep)')
    parser.add_argument('--causal_quantile', type=float, default=0.50, help='Causal quantile (default: 0.50)')

    # Meta arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default='tep', help='Dataset name (default: tep)')

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
    parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs (default: 3000)')
    parser.add_argument('--hidden_layer_size', type=int, default=64, help='Hidden layer size (default: 64)')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--recon_threshold', type=float, default=0.95, help='Reconstruction threshold (default: 0.95)')
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.99, 
                       help='Root cause threshold for encoder (default: 0.99)')
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.99, 
                       help='Root cause threshold for decoder (default: 0.99)')
    parser.add_argument('--training_aerca', type=int, default=1, help='Flag for training AERCA (default: 1)')
    parser.add_argument('--initial_z_score', type=float, default=3.0, help='Initial Z-score (default: 3.0)')
    parser.add_argument('--risk', type=float, default=1e-2, help='Risk (default: 1e-2)')
    parser.add_argument('--initial_level', type=float, default=0.98, help='Initial level (default: 0.98)')
    parser.add_argument('--num_candidates', type=int, default=100, help='Number of candidates (default: 100)')

    return parser

if __name__ == "__main__":
    try:
        arg_parser = create_arg_parser()
        args = arg_parser.parse_args()
        print(args)
    except Exception as e:
        print(f"Error parsing arguments: {e}")
