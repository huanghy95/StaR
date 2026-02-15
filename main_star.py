"""
Main script for running StaR (Stateful Dynamic-Graph Root Cause Analysis)

This script supports linear StaR with comprehensive training and testing phases
including causal discovery and root cause analysis.

Usage:
    # Linear StaR-GC (default)
    python main_star.py --dataset_name linear
    python main_star.py --dataset_name lorenz96
    python main_star.py --dataset_name swat
    python main_star.py --dataset_name temporal_service_dynamic
    python main_star.py --dataset_name random_connection
    
    # Ablation Study: Disable Message Passing (test effect of message passing component)
    python main_star.py --dataset_name lorenz96 --disable_message_passing
    python main_star.py --dataset_name temporal_service_dynamic --disable_message_passing
    
    # Ablation Study: Different Memory Mechanisms
    python main_star.py --dataset_name linear --memory_updater_type lstm
    python main_star.py --dataset_name linear --memory_updater_type transformer
    python main_star.py --dataset_name linear --memory_updater_type mlp
    
    # Ablation Study: Different Memory/Time/Message Dimensions
    python main_star.py --dataset_name linear --memory_dim 32
    python main_star.py --dataset_name linear --time_dim 16
    python main_star.py --dataset_name linear --message_dim 32
"""

import sys
import os
import logging
import argparse
import json
import time
from datetime import datetime

from datasets import linear, lotka_volterra, lorenz96, swat, nonlinear, msds, temporal_service_dynamic, random_connection_service, tep, msl, smd, aiops
from args import linear_args, lotka_volterra_args, lorenz96_args, swat_args, msds_args, nonlinear_args, temporal_service_dynamic_args, random_connection_args, tep_args, msl_args, smd_args, aiops_args
from models import star
from utils import utils
import warnings
warnings.filterwarnings("ignore")


def run_single_experiment(dataset, options, pre_args, mapping):
    """
    Run a single experiment with linear Granger causality.
    
    Returns:
        dict: Results dictionary with performance metrics and timing information
    """
    print(f"\n{'='*80}")
    print(f"Running StaR with Linear Granger Causality")
    print(f"{'='*80}")
    print(f"Memory Dimension: {pre_args.memory_dim}")
    print(f"Time Dimension: {pre_args.time_dim}")
    print(f"Message Dimension: {pre_args.message_dim}")
    print(f"Memory Updater Type: {pre_args.memory_updater_type.upper()}")
    print(f"Message Passing: {'DISABLED (Ablation Study)' if pre_args.disable_message_passing else 'ENABLED'}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Get the original dataset name (without timestamp)
    original_dataset_name = options.get('data_name', options.get('dataset_name', 'unknown'))
    
    # Create timestamped data name for model naming only
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_data_name = f"{original_dataset_name}_{timestamp}"
    
    # Instantiate StaR model
    print("Using Linear StaR-GC")
    star_model = star.StaR(
        num_vars=dataset.num_vars,
        hidden_layer_size=options['hidden_layer_size'],
        num_hidden_layers=options['num_hidden_layers'],
        device=options['device'],
        window_size=options['window_size'],
        stride=options['stride'],
        encoder_alpha=options.get('encoder_alpha', 0.5),
        decoder_alpha=options.get('decoder_alpha', 0.5),
        encoder_gamma=options['encoder_gamma'],
        decoder_gamma=options['decoder_gamma'],
        encoder_lambda=options['encoder_lambda'],
        decoder_lambda=options['decoder_lambda'],
        beta=options['beta'],
        lr=options['lr'],
        epochs=options['epochs'],
        recon_threshold=options['recon_threshold'],
        data_name=timestamped_data_name,
        causal_quantile=options['causal_quantile'],
        root_cause_threshold_encoder=options['root_cause_threshold_encoder'],
        root_cause_threshold_decoder=options['root_cause_threshold_decoder'],
        initial_z_score=options.get('initial_z_score', 3.0),
        risk=options['risk'],
        initial_level=options['initial_level'],
        num_candidates=options['num_candidates'],
        memory_dim=pre_args.memory_dim,
        time_dim=pre_args.time_dim,
        message_dim=pre_args.message_dim,
        granger_method='linear',
        disable_message_passing=pre_args.disable_message_passing,
        memory_updater_type=pre_args.memory_updater_type,
        num_attention_heads=pre_args.num_attention_heads,
    )
    
    results = {
        'method': 'linear',
        'start_time': datetime.now().isoformat(),
        'model_id': getattr(star_model, 'model_id', None),
        'model_dir': getattr(star_model, 'model_dir', None)
    }
    
    # Train and evaluate the model
    print(f'\nStarting StaR Training and Evaluation (LINEAR)')
    print('='*60)
    
    try:
        # Prepare training data based on dataset configuration
        if mapping["use_slice"]:
            training_data = dataset.data_dict['x_n_list'][:options['training_size']]
        else:
            training_data = dataset.data_dict['x_n_list']
        
        # Training phase
        print('Start training StaR model...')
        star_model._training(training_data)
        print('Done training')
        
        # Testing phase for causal discovery (applies only if slicing is used)
        causal_results = None
        root_cause_results = None
        
        if mapping["use_slice"]:
            test_causal = dataset.data_dict['x_n_list'][options['training_size']:]
            print('Start testing StaR model for causal discovery...')
            causal_results = star_model._testing_causal_discover(test_causal, dataset.data_dict['causal_struct'])
            print('Done testing for causal discovery')
            
            # Testing phase for root cause analysis (with slicing)
            if 'x_ab_list' in dataset.data_dict and 'label_list' in dataset.data_dict:
                test_root_cause_data = dataset.data_dict['x_ab_list'][options['training_size']:]
                test_labels = dataset.data_dict['label_list'][options['training_size']:]
                connection_history_list = dataset.data_dict.get('connection_history_list')
                timing_info_list = dataset.data_dict.get('timing_info')
                if connection_history_list is not None:
                    connection_history_list = connection_history_list[options['training_size']:]
                    timing_info_list = timing_info_list[options['training_size']:]
                print('Start testing StaR model for root cause analysis...')
                root_cause_results = star_model._testing_root_cause(test_root_cause_data, test_labels, 
                                                                         connection_history_list, timing_info_list)
                print('Done testing for root cause analysis')
        else:
            # Testing phase for root cause analysis (without slicing - for MSDS and SWaT datasets)
            if 'x_ab_list' in dataset.data_dict and 'label_list' in dataset.data_dict:
                test_root_cause_data = dataset.data_dict['x_ab_list']
                test_labels = dataset.data_dict['label_list']
                connection_history_list = dataset.data_dict.get('connection_history_list')
                timing_info_list = dataset.data_dict.get('timing_info')
                print('Start testing StaR model for root cause analysis...')
                root_cause_results = star_model._testing_root_cause(test_root_cause_data, test_labels,
                                                                         connection_history_list, timing_info_list)
                print('Done testing for root cause analysis')
        
        total_time = time.time() - start_time
        
        # Combine results
        results = {
            'granger_method': 'linear',
            'total_time_seconds': total_time,
            'memory_dim': pre_args.memory_dim,
            'time_dim': pre_args.time_dim,
            'message_dim': pre_args.message_dim,
            'memory_updater_type': pre_args.memory_updater_type,
            'disable_message_passing': pre_args.disable_message_passing,
            'dataset_name': options.get('dataset_name', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'model_save_dir': getattr(star_model, 'save_dir', None),
            'model_name': getattr(star_model, 'model_name', None)
        }
        
        # Add testing results if available
        if causal_results:
            results['causal_discovery'] = causal_results
        if root_cause_results:
            results['root_cause_analysis'] = root_cause_results
        
        print('='*60)
        print(f'StaR Experiment Complete ({total_time:.2f}s)')
        print('='*60 + '\n')
        
        return results
        
    except Exception as e:
        print(f"Error during training/evaluation: {str(e)}")
        raise e


def save_experiment_results(results_list, dataset_name):
    """Save experiment results to JSON file."""
    results_dir = os.path.join(os.getcwd(), "experiment_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aerca_tgn_{dataset_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\nExperiment results saved to: {filepath}")
    return filepath


def main(argv):
    """
    Main function to run the StaR model on a specified dataset.

    The script supports multiple datasets. It selects the appropriate dataset class,
    argument parser, and log file based on the provided --dataset_name argument.
    If preprocessing_data is set to 1, the dataset is generated and saved; otherwise,
    the existing data is loaded.

    Args:
        argv (list): List of command-line arguments.
    """
    # Preliminary parsing: retrieve the dataset name and TGN configuration.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--dataset_name',
        type=str,
        default='linear',
        help='Name of the dataset to run. Options: linear, lotka_volterra, lorenz96, msds, swat, nonlinear, temporal_service_dynamic, random_connection, tep, msl, smd, aiops'
    )
    pre_parser.add_argument(
        '--memory_dim',
        type=int,
        default=64,
        help='Dimension of TGN memory vectors'
    )
    pre_parser.add_argument(
        '--time_dim',
        type=int,
        default=32,
        help='Dimension of temporal encoding in TGN'
    )
    pre_parser.add_argument(
        '--message_dim',
        type=int,
        default=64,
        help='Dimension of messages in TGN'
    )
    pre_parser.add_argument(
        '--disable_message_passing',
        action='store_true',
        help='Disable message passing in TGN (ablation study: test effect of message passing component)'
    )
    pre_parser.add_argument(
        '--num_attention_heads',
        type=int,
        default=4,
        help='Number of attention heads for transformer memory updater'
    )
    pre_parser.add_argument(
        '--memory_updater_type',
        type=str,
        default='gru',
        choices=['gru', 'lstm', 'transformer', 'mlp'],
        help='Type of memory updater mechanism for TGN (ablation study)'
    )
    
    pre_args, remaining_args = pre_parser.parse_known_args(argv[1:])
    dataset_name = pre_args.dataset_name.lower()

    # Map dataset names to their configuration
    dataset_mapping = {
        "linear": {
            "args": linear_args.create_arg_parser,
            "dataset_class": linear.Linear,
            "log_file": "linear_tgn.log",
            "use_slice": True
        },
        "lotka_volterra": {
            "args": lotka_volterra_args.create_arg_parser,
            "dataset_class": lotka_volterra.LotkaVolterra,
            "log_file": "lotka_volterra_tgn.log",
            "use_slice": True
        },
        "lorenz96": {
            "args": lorenz96_args.create_arg_parser,
            "dataset_class": lorenz96.Lorenz96,
            "log_file": "lorenz96_tgn.log",
            "use_slice": True
        },
        "msds": {
            "args": msds_args.create_arg_parser,
            "dataset_class": msds.MSDS,
            "log_file": "msds_tgn.log",
            "use_slice": False
        },
        "swat": {
            "args": swat_args.create_arg_parser,
            "dataset_class": swat.SWaT,
            "log_file": "swat_tgn.log",
            "use_slice": False
        },
        "nonlinear": {
            "args": nonlinear_args.create_arg_parser,
            "dataset_class": nonlinear.Nonlinear,
            "log_file": "nonlinear_tgn.log",
            "use_slice": True
        },
        "temporal_service_dynamic": {
            "args": temporal_service_dynamic_args.create_arg_parser,
            "dataset_class": temporal_service_dynamic.DynamicTemporalService,
            "log_file": "temporal_service_dynamic_tgn.log",
            "use_slice": True
        },
        "random_connection": {
            "args": random_connection_args.create_arg_parser,
            "dataset_class": random_connection_service.RandomConnectionService,
            "log_file": "random_connection_tgn.log",
            "use_slice": True
        },
        "tep": {
            "args": tep_args.create_arg_parser,
            "dataset_class": tep.TEP,
            "log_file": "tep_tgn.log",
            "use_slice": False
        },
        "msl": {
            "args": msl_args.create_arg_parser,
            "dataset_class": msl.MSL,
            "log_file": "msl_tgn.log",
            "use_slice": False
        },
        "smd": {
            "args": smd_args.create_arg_parser,
            "dataset_class": smd.SMD,
            "log_file": "smd_tgn.log",
            "use_slice": False
        },
        "aiops": {
            "args": aiops_args.create_arg_parser,
            "dataset_class": aiops.AIOps,
            "log_file": "aiops_tgn.log",
            "use_slice": False
        }
    }

    # Ensure the specified dataset is recognized.
    if dataset_name not in dataset_mapping:
        print("Dataset '{}' not recognized. Available options are: {}"
              .format(dataset_name, list(dataset_mapping.keys())))
        sys.exit(1)

    mapping = dataset_mapping[dataset_name]

    # Set up logging
    logging_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    log_file_path = os.path.join(logging_dir, mapping["log_file"])
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse remaining command-line arguments
    parser = mapping["args"]()
    args, unknown = parser.parse_known_args(remaining_args)
    options = vars(args)

    # Set random seed for reproducibility
    utils.set_seed(options['seed'])
    print('Set seed: {}'.format(options['seed']))

    # Get dataset class (not instantiated yet)
    dataset_class = mapping["dataset_class"]
    
    # Create a dataset instance for preprocessing/loading data
    data_instance = dataset_class(options)
    if options['preprocessing_data'] == 1:
        print('Preprocessing data: generating and saving new data...')
        data_instance.generate_example()
        data_instance.save_data()
    else:
        print('Loading existing data...')
        data_instance.load_data()

    print(f"\n{'='*80}")
    print("StaR Experiment Configuration")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Model Type: Linear StaR-GC")
    print(f"Memory Dimension: {pre_args.memory_dim}")
    print(f"Time Dimension: {pre_args.time_dim}")
    print(f"Message Dimension: {pre_args.message_dim}")
    print(f"Memory Updater: {pre_args.memory_updater_type.upper()}")
    print(f"Message Passing: {'DISABLED' if pre_args.disable_message_passing else 'ENABLED'}")
    print(f"{'='*80}\n")
    
    # Run experiment
    try:
        result = run_single_experiment(data_instance, options, pre_args, mapping)
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        raise e
    
    print(f"\n{'='*80}")
    print('All Tasks Complete!')
    print(f"{'='*80}")


if __name__ == '__main__':
    main(sys.argv)
