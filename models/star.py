"""
StaR: Root Cause Analysis using Temporal Graph Networks with Granger Causality

This module extends the original AERCA framework by replacing SENN with StaR-GC,
enabling better temporal modeling and graph-based reasoning for RCA tasks.
"""

import os
import json
import hashlib
from models.star_gc import StaRGC
from models.star_gc_flexible import StaRGC_Flexible
import torch.nn as nn
import torch
from utils.utils import (compute_kl_divergence, sliding_window_view_torch,
                         eval_causal_structure, eval_causal_structure_binary,
                         pot, topk, topk_at_step)
from utils.model_manager import ModelManager
from numpy.lib.stride_tricks import sliding_window_view
import logging
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


class StaR(nn.Module):
    """
    AERCA with Temporal Graph Networks (StaR).
    
    This model combines TGN's temporal modeling capabilities with Granger causality
    for robust root cause analysis in multivariate time series.
    
    Key improvements over original AERCA:
    1. Temporal memory to track node states over time
    2. Message passing between nodes for better dependency modeling
    3. Enhanced causal discovery through graph structure awareness
    """
    def __init__(self, num_vars: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 window_size: int, stride: int = 1, encoder_alpha: float = 0.5, decoder_alpha: float = 0.5,
                 encoder_gamma: float = 0.5, decoder_gamma: float = 0.5,
                 encoder_lambda: float = 0.5, decoder_lambda: float = 0.5,
                 beta: float = 0.5, lr: float = 0.0001, epochs: int = 100,
                 recon_threshold: float = 0.95, data_name: str = 'ld',
                 causal_quantile: float = 0.80, root_cause_threshold_encoder: float = 0.95,
                 root_cause_threshold_decoder: float = 0.95, initial_z_score: float = 3.0,
                 risk: float = 1e-2, initial_level: float = 0.98, num_candidates: int = 100,
                 memory_dim: int = 64, time_dim: int = 32, message_dim: int = 64,
                 granger_method: str = 'linear', disable_message_passing: bool = False,
                 memory_updater_type: str = 'gru', num_attention_heads: int = 4):
        """
        Args:
            num_vars: Number of variables in the system
            hidden_layer_size: Size of hidden layers
            num_hidden_layers: Number of hidden layers (kept for compatibility)
            device: Torch device
            window_size: Size of sliding window
            stride: Stride for sliding window
            encoder_alpha: L1/L2 trade-off for encoder sparsity
            decoder_alpha: L1/L2 trade-off for decoder sparsity
            encoder_gamma: Smoothness weight for encoder
            decoder_gamma: Smoothness weight for decoder
            encoder_lambda: Sparsity weight for encoder
            decoder_lambda: Sparsity weight for decoder
            beta: KL divergence weight
            lr: Learning rate
            epochs: Number of training epochs
            recon_threshold: Reconstruction threshold quantile
            data_name: Dataset name for model saving
            causal_quantile: Quantile for causal graph thresholding
            root_cause_threshold_encoder: Threshold for encoder-based root cause
            root_cause_threshold_decoder: Threshold for decoder-based root cause
            initial_z_score: Initial z-score for POT
            risk: Risk parameter for POT
            initial_level: Initial level for POT
            num_candidates: Number of candidates for POT
            memory_dim: Dimension of TGN memory vectors
            time_dim: Dimension of temporal encoding
            message_dim: Dimension of messages in TGN
            granger_method: Granger causality method ('linear')
            disable_message_passing: If True, bypass message passing (ablation study)
            memory_updater_type: Type of memory updater ('gru', 'lstm', 'transformer', 'mlp')
            num_attention_heads: Number of attention heads for transformer updater
        """
        super(StaR, self).__init__()
        
        # TGN-based encoder and decoders with method selection
        # Use flexible version if non-GRU memory updater is requested
        if memory_updater_type != 'gru':
            encoder_class = StaRGC_Flexible
            decoder_class = StaRGC_Flexible
            kwargs = {
                'disable_message_passing': disable_message_passing,
                'memory_updater_type': memory_updater_type,
                'num_attention_heads': num_attention_heads
            }
        else:
            encoder_class = StaRGC
            decoder_class = StaRGC
            kwargs = {'disable_message_passing': disable_message_passing}
        
        # Instantiate the models with num_hidden_layers parameter
        common_args = (num_vars, window_size, hidden_layer_size, 
                      memory_dim, time_dim, message_dim, device, num_hidden_layers)
        
        self.encoder = encoder_class(*common_args, **kwargs)
        self.decoder = decoder_class(*common_args, **kwargs)
        self.decoder_prev = decoder_class(*common_args, **kwargs)
        
        self.device = device
        self.num_vars = num_vars
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.stride = stride
        self.encoder_alpha = encoder_alpha
        self.decoder_alpha = decoder_alpha
        self.encoder_gamma = encoder_gamma
        self.decoder_gamma = decoder_gamma
        self.encoder_lambda = encoder_lambda
        self.decoder_lambda = decoder_lambda
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.recon_threshold = recon_threshold
        self.root_cause_threshold_encoder = root_cause_threshold_encoder
        self.root_cause_threshold_decoder = root_cause_threshold_decoder
        self.initial_z_score = initial_z_score
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_reduction = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)
        
        # Model naming for saving (shortened to avoid filesystem limits)
        # Create a short unique identifier based on parameters
        param_dict = {
            'data_name': data_name,
            'window_size': window_size,
            'stride': stride,
            'encoder_alpha': encoder_alpha,
            'decoder_alpha': decoder_alpha,
            'encoder_gamma': encoder_gamma,
            'decoder_gamma': decoder_gamma,
            'encoder_lambda': encoder_lambda,
            'decoder_lambda': decoder_lambda,
            'beta': beta,
            'lr': lr,
            'epochs': epochs,
            'hidden_layer_size': hidden_layer_size,
            'memory_dim': memory_dim,
            'time_dim': time_dim,
            'message_dim': message_dim,
            'disable_message_passing': disable_message_passing,
            'memory_updater_type': memory_updater_type
        }
        
        # Create hash of parameters for unique identification
        param_str = json.dumps(param_dict, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Short model name with hash
        ablation_suffix = '_NoMP' if disable_message_passing else ''
        memory_suffix = f'_{memory_updater_type.upper()}' if memory_updater_type != 'gru' else ''
        self.model_name = f'StaR_{data_name}_{param_hash}{memory_suffix}{ablation_suffix}'
        self.model_params = param_dict
        
        self.causal_quantile = causal_quantile
        self.risk = risk
        self.initial_level = initial_level
        self.num_candidates = num_candidates
        
        # Initialize model manager for organized model saving
        self.model_manager = ModelManager()
        # Extract original dataset name from timestamped data_name for folder structure
        self.dataset_name = data_name.split('_')[0] if '_' in data_name else data_name
        # Use Granger method for naming, with ablation suffix
        self.method_name = f"{granger_method}{ablation_suffix}"
        self.model_id = None
        self.model_dir = None

    def _setup_model_directory(self):
        """Setup model directory using ModelManager."""
        if self.model_dir is None:
            # Create model directory with proper structure
            self.model_id, self.model_dir = self.model_manager.create_model_directory(
                self.dataset_name, self.method_name, self.model_params
            )
            print(f"Model directory created: {self.model_dir}")
            print(f"Model ID: {self.model_id}")

    def _log_and_print(self, msg, *args):
        """Helper method to log and print testing results."""
        final_msg = msg.format(*args) if args else msg
        logging.info(final_msg)
        print(final_msg)

    def _sparsity_loss(self, coeffs, alpha):
        """Compute sparsity loss as weighted combination of L1 and L2 norms."""
        norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
        norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))
        return (1 - alpha) * norm2 + alpha * norm1

    def _smoothness_loss(self, coeffs):
        """Compute temporal smoothness loss."""
        return torch.norm(coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :], dim=1).mean()

    def encoding(self, xs, adjacency_mask=None):
        """
        Encoding phase: Transform input to latent exogenous variables.
        
        Args:
            xs: Input time series
            adjacency_mask: Optional adjacency mask for dynamic graph inference
        Returns:
            us: Exogenous variables (prediction residuals)
            coeffs: Encoder causal coefficients
            nexts: Ground truth next-step values
            winds: Historical windows
        """
        windows = sliding_window_view(xs, (self.window_size + 1, self.num_vars))[:, 0, :, :]
        winds = windows[:, :-1, :]
        nexts = windows[:, -1, :]
        winds = torch.tensor(winds).float().to(self.device)
        nexts = torch.tensor(nexts).float().to(self.device)
        
        # Adjust adjacency mask to match winds shape if provided
        encoder_mask = None
        if adjacency_mask is not None:
            encoder_mask = adjacency_mask[:winds.shape[0]] if adjacency_mask.shape[0] > winds.shape[0] else adjacency_mask
        
        # Use TGN encoder with adjacency mask
        preds, coeffs = self.encoder(winds, adjacency_mask=encoder_mask)
        us = preds - nexts
        
        return us, coeffs, nexts[self.window_size:], winds[:-self.window_size]

    def decoding(self, us, winds, add_u=True, adjacency_mask=None):
        """
        Decoding phase: Reconstruct observations from exogenous variables.
        
        Args:
            us: Exogenous variables
            winds: Historical windows
            add_u: Whether to add exogenous variables
            adjacency_mask: Optional adjacency mask for dynamic graph inference
        Returns:
            nexts_hat: Reconstructed next-step predictions
            coeffs: Decoder causal coefficients
            prev_coeffs: Previous decoder causal coefficients
        """
        u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = u_windows[:, :-1, :]
        u_next = u_windows[:, -1, :]

        # Adjust adjacency mask for decoder phases
        decoder_mask = None
        prev_mask = None
        if adjacency_mask is not None:
            if adjacency_mask.shape[0] >= u_winds.shape[0]:
                decoder_mask = adjacency_mask[self.window_size:self.window_size + u_winds.shape[0]]
            if adjacency_mask.shape[0] >= winds.shape[0]:
                prev_mask = adjacency_mask[:winds.shape[0]]
        
        # Use TGN decoders with adjacency mask
        preds, coeffs = self.decoder(u_winds, adjacency_mask=decoder_mask)
        prev_preds, prev_coeffs = self.decoder_prev(winds, adjacency_mask=prev_mask)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds
            
        return nexts_hat, coeffs, prev_coeffs

    def forward(self, x, add_u=True, adjacency_mask=None):
        """
        Forward pass through StaR.
        
        Args:
            x: Input time series
            add_u: Whether to add exogenous variables in decoding
            adjacency_mask: Optional adjacency mask of shape (num_windows, num_vars, num_vars)
                          to mask connections during inference. Used for dynamic graph datasets.
        Returns:
            nexts_hat: Reconstructed predictions
            nexts: Ground truth
            encoder_coeffs: Encoder causal coefficients
            decoder_coeffs: Decoder causal coefficients
            prev_coeffs: Previous decoder coefficients
            kl_div: KL divergence
            us: Exogenous variables
        """
        us, encoder_coeffs, nexts, winds = self.encoding(x, adjacency_mask=adjacency_mask)
        kl_div = compute_kl_divergence(us, self.device)
        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u, adjacency_mask=adjacency_mask)
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us

    def _training_step(self, x, add_u=True):
        """Single training step."""
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)
        
        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
        logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
        logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
        logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        loss_prev_smooth = self._smoothness_loss(prev_coeffs)
        logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl)
        logging.info('Total loss: %s', loss.item())

        return loss

    def _training(self, xs):
        """Training loop with validation and early stopping."""
        # Setup model directory for saving
        self._setup_model_directory()
        
        # Reset TGN memories at start of training
        self.encoder.reset_memory()
        self.decoder.reset_memory()
        self.decoder_prev.reset_memory()
        
        if len(xs) == 1:
            xs_train = xs[:, :int(0.8 * len(xs[0]))]
            xs_val = xs[:, int(0.8 * len(xs[0])):]
        else:
            xs_train = xs[:int(0.8 * len(xs))]
            xs_val = xs[int(0.8 * len(xs)):]
            
        best_val_loss = np.inf
        count = 0
        
        for epoch in tqdm(range(self.epochs), desc=f'Epoch'):
            count += 1
            epoch_loss = 0
            self.train()
            
            for x in xs_train:
                self.optimizer.zero_grad()
                loss = self._training_step(x)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
            logging.info('Epoch %s/%s', epoch + 1, self.epochs)
            logging.info('Epoch training loss: %s', epoch_loss)
            logging.info('-------------------')
            
            epoch_val_loss = 0
            self.eval()
            with torch.no_grad():
                for x in xs_val:
                    loss = self._training_step(x)
                    epoch_val_loss += loss.item()
                    
            logging.info('Epoch val loss: %s', epoch_val_loss)
            logging.info('-------------------')
            
            if epoch_val_loss < best_val_loss:
                count = 0
                logging.info(f'Saving model at epoch {epoch + 1}')
                logging.info(f'Saving model name: {self.model_name}.pt')
                best_val_loss = epoch_val_loss
                
                # Save model using ModelManager
                self.model_manager.save_model_state(self.model_dir, self.state_dict())
                
            if count >= 200:
                print('Early stopping')
                break
                
        # Load model using ModelManager
        model_state = self.model_manager.load_model_state(self.model_dir, self.device)
        self.load_state_dict(model_state)
        logging.info('Training complete')
        self._get_recon_threshold(xs_val)
        self._get_root_cause_threshold_encoder(xs_val)
        self._get_root_cause_threshold_decoder(xs_val)

    def _testing_step(self, x, label=None, add_u=True, adjacency_mask=None):
        """
        Single testing step.
        
        Args:
            x: Input time series
            label: Optional labels
            add_u: Whether to add exogenous variables
            adjacency_mask: Optional adjacency mask for dynamic graph inference
        """
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u, adjacency_mask=adjacency_mask)

        if label is not None:
            preprocessed_label = sliding_window_view(label, (self.window_size + 1, self.num_vars))[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None

        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
        logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
        logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
        logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        loss_prev_smooth = self._smoothness_loss(prev_coeffs)
        logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl)
        logging.info('Total loss: %s', loss.item())

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us

    def _get_recon_threshold(self, xs):
        """Compute reconstruction threshold on validation data."""
        self.eval()
        losses_list = []
        with torch.no_grad():
            for x in xs:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x, add_u=False)
                loss_arr = self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy().ravel()
                losses_list.append(loss_arr)
        recon_losses = np.concatenate(losses_list)
        self.recon_threshold_value = np.quantile(recon_losses, self.recon_threshold)
        self.recon_mean = np.mean(recon_losses)
        self.recon_std = np.std(recon_losses)
        # Save thresholds using ModelManager
        self.model_manager.save_threshold(self.model_dir, 'recon_threshold', self.recon_threshold_value)
        self.model_manager.save_threshold(self.model_dir, 'recon_mean', self.recon_mean)
        self.model_manager.save_threshold(self.model_dir, 'recon_std', self.recon_std)

    def _get_root_cause_threshold_encoder(self, xs):
        """Compute encoder-based root cause thresholds."""
        self.eval()
        us_list = []
        with torch.no_grad():
            for x in xs:
                us = self._testing_step(x)[-1]
                us_list.append(us.cpu().numpy())
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self.lower_encoder = np.quantile(us_all, (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.upper_encoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0)
        # Save encoder thresholds using ModelManager
        self.model_manager.save_threshold(self.model_dir, 'lower_encoder', self.lower_encoder)
        self.model_manager.save_threshold(self.model_dir, 'upper_encoder', self.upper_encoder)
        self.model_manager.save_threshold(self.model_dir, 'us_mean_encoder', self.us_mean_encoder)
        self.model_manager.save_threshold(self.model_dir, 'us_std_encoder', self.us_std_encoder)

    def _get_root_cause_threshold_decoder(self, xs):
        """Compute decoder-based root cause thresholds."""
        self.eval()
        diff_list = []
        with torch.no_grad():
            for x in xs:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x, add_u=False)
                diff = (nexts - nexts_hat).cpu().numpy().ravel()
                diff_list.append(diff)
        us_all = np.concatenate(diff_list, axis=0).reshape(-1, self.num_vars)
        self.lower_decoder = np.quantile(us_all, (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.upper_decoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.us_mean_decoder = np.mean(us_all, axis=0)
        self.us_std_decoder = np.std(us_all, axis=0)
        # Save decoder thresholds using ModelManager
        self.model_manager.save_threshold(self.model_dir, 'lower_decoder', self.lower_decoder)
        self.model_manager.save_threshold(self.model_dir, 'upper_decoder', self.upper_decoder)
        self.model_manager.save_threshold(self.model_dir, 'us_mean_decoder', self.us_mean_decoder)
        self.model_manager.save_threshold(self.model_dir, 'us_std_decoder', self.us_std_decoder)

    def _testing_root_cause(self, xs, labels, connection_history_list=None, timing_info_list=None):
        """Test root cause analysis performance."""
        # Load model and encoder parameters using ModelManager
        model_state = self.model_manager.load_model_state(self.model_dir, self.device)
        self.load_state_dict(model_state)
        self.eval()
        self.us_mean_encoder = self.model_manager.load_threshold(self.model_dir, 'us_mean_encoder')
        self.us_std_encoder = self.model_manager.load_threshold(self.model_dir, 'us_std_encoder')

        # Check if we have dynamic graph data
        has_dynamic_graph = connection_history_list is not None and len(connection_history_list) > 0
        
        # If dynamic graph available, compute both static and dynamic versions
        if has_dynamic_graph:
            self._log_and_print('=' * 50)
            self._log_and_print('Computing AC@k for STATIC graph (baseline):')
            self._compute_root_cause_metrics(xs, labels, connection_history_list=None, 
                                            timing_info_list=timing_info_list, mode='static')
            
            self._log_and_print('=' * 50)
            self._log_and_print('Computing AC@k for DYNAMIC graph (with connection_history):')
            self._compute_root_cause_metrics(xs, labels, connection_history_list=connection_history_list,
                                            timing_info_list=timing_info_list, mode='dynamic')
        else:
            # No dynamic graph data, just compute static version
            self._compute_root_cause_metrics(xs, labels, connection_history_list=None,
                                            timing_info_list=timing_info_list, mode='static')
    
    def _compute_root_cause_metrics(self, xs, labels, connection_history_list=None, 
                                     timing_info_list=None, mode='static'):
        """
        Compute root cause analysis metrics.
        
        Args:
            xs: List of time series samples
            labels: List of labels
            connection_history_list: Optional dynamic graph structure
            timing_info_list: Optional timing information
            mode: 'static' or 'dynamic' (for logging purposes)
        """
        # Reset memories for testing
        self.encoder.reset_memory()
        self.decoder.reset_memory()
        self.decoder_prev.reset_memory()

        # Collect latent representations
        us_list = []
        us_sample_list = []
        z_scores_list = []  # For GAAC metrics (with mask)
        z_scores_no_mask_list = []  # For TDS metric (without mask)
        
        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                
                # Prepare adjacency mask from connection history if available
                adjacency_mask = None
                if connection_history_list is not None and len(connection_history_list) > i:
                    conn_history = connection_history_list[i]  # Shape: (T-1, num_vars, num_vars)
                    if conn_history is not None and len(conn_history) > 0:
                        # Create windows to match the sliding window view
                        T = x.shape[0]
                        num_windows = T - self.window_size
                        
                        mask_list = []
                        for w in range(num_windows):
                            conn_idx = min(w + self.window_size - 1, len(conn_history) - 1)
                            mask_list.append(conn_history[conn_idx])
                        
                        if mask_list:
                            adjacency_mask = torch.tensor(np.stack(mask_list), dtype=torch.float32).to(self.device)
                
                # Compute WITH mask for AC@k and GAAC
                us = self._testing_step(x, label, add_u=False, adjacency_mask=adjacency_mask)[-1]
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())
                
                # Compute WITHOUT mask for TDS (to test true temporal memory)
                if connection_history_list is not None:
                    us_no_mask = self._testing_step(x, label, add_u=False, adjacency_mask=None)[-1]
                    z_scores_no_mask = (-(us_no_mask[self.window_size:].cpu().numpy() - self.us_mean_encoder) / self.us_std_encoder)
                    z_scores_no_mask_list.append(z_scores_no_mask)

        # Combine all latent representations for POT threshold computation
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self._log_and_print('=' * 50)
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        us_all_z_score_pot = []
        for i in range(self.num_vars):
            pot_val, _ = pot(us_all_z_score[:, i], self.risk, self.initial_level, self.num_candidates)
            us_all_z_score_pot.append(pot_val)
        us_all_z_score_pot = np.array(us_all_z_score_pot)

        # Compute top-k statistics
        k_all = []
        k_at_step_all = []
        for i in range(len(xs)):
            us_sample = us_sample_list[i]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            z_scores_list.append(z_scores)  # Store for GAAC
            k_lst = topk(z_scores, labels[i][self.window_size * 2:], us_all_z_score_pot)
            k_at_step = topk_at_step(z_scores, labels[i][self.window_size * 2:])
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)
        # Convert to numpy arrays for mean and std computation
        k_all_array = np.array(k_all)  # Shape: (n_samples, k_range)
        k_at_step_all_array = np.array(k_at_step_all)  # Shape: (n_samples, k_range)
        # Compute mean and std across samples
        k_all_mean = k_all_array.mean(axis=0)
        k_all_std = k_all_array.std(axis=0)
        k_at_step_all_mean = k_at_step_all_array.mean(axis=0)
        k_at_step_all_std = k_at_step_all_array.std(axis=0)
        
        # AC@K metrics (mean +/- std)
        ac_at_mean = [k_at_step_all_mean[0], k_at_step_all_mean[2], k_at_step_all_mean[4], k_at_step_all_mean[9]]
        ac_at_std = [k_at_step_all_std[0], k_at_step_all_std[2], k_at_step_all_std[4], k_at_step_all_std[9]]
        self._log_and_print('[{}] Root cause analysis AC@1: {:.5f}+/-{:.5f}', mode.upper(), ac_at_mean[0], ac_at_std[0])
        self._log_and_print('[{}] Root cause analysis AC@3: {:.5f}+/-{:.5f}', mode.upper(), ac_at_mean[1], ac_at_std[1])
        self._log_and_print('[{}] Root cause analysis AC@5: {:.5f}+/-{:.5f}', mode.upper(), ac_at_mean[2], ac_at_std[2])
        self._log_and_print('[{}] Root cause analysis AC@10: {:.5f}+/-{:.5f}', mode.upper(), ac_at_mean[3], ac_at_std[3])
        avg_at_10_mean = np.mean(k_at_step_all_mean)
        avg_at_10_std = np.std(k_at_step_all_array.mean(axis=1))  # Std of per-sample means
        self._log_and_print('[{}] Root cause analysis Avg@10: {:.5f}+/-{:.5f}', mode.upper(), avg_at_10_mean, avg_at_10_std)

        # AC*@K metrics (mean +/- std)
        ac_star_at_mean = [k_all_mean[0], k_all_mean[9], k_all_mean[99], k_all_mean[499]]
        ac_star_at_std = [k_all_std[0], k_all_std[9], k_all_std[99], k_all_std[499]]
        self._log_and_print('[{}] Root cause analysis AC*@1: {:.5f}+/-{:.5f}', mode.upper(), ac_star_at_mean[0], ac_star_at_std[0])
        self._log_and_print('[{}] Root cause analysis AC*@10: {:.5f}+/-{:.5f}', mode.upper(), ac_star_at_mean[1], ac_star_at_std[1])
        self._log_and_print('[{}] Root cause analysis AC*@100: {:.5f}+/-{:.5f}', mode.upper(), ac_star_at_mean[2], ac_star_at_std[2])
        self._log_and_print('[{}] Root cause analysis AC*@500: {:.5f}+/-{:.5f}', mode.upper(), ac_star_at_mean[3], ac_star_at_std[3])
        avg_star_500_mean = np.mean(k_all_mean)
        avg_star_500_std = np.std(k_all_array.mean(axis=1))  # Std of per-sample means
        self._log_and_print('[{}] Root cause analysis Avg*@500: {:.5f}+/-{:.5f}', mode.upper(), avg_star_500_mean, avg_star_500_std)
        
        # Compute dynamic graph metrics
        if connection_history_list is not None and timing_info_list is not None:
            from utils.utils import (
                compute_gaac,
                prepare_scores_for_metrics
            )
            from utils.tds_metric import compute_temporal_decoupling_score, interpret_tds
            
            # Compute GAAC (original metric for comparison)
            gaac_score = compute_gaac(
                z_scores_list=z_scores_list,
                connection_history_list=connection_history_list,
                labels_list=labels,
                timing_info_list=timing_info_list,
                window_size=self.window_size
            )
            self._log_and_print('=' * 50)
            self._log_and_print('Graph-Aware Anomaly Contextualization (GAAC): {:.5f}', gaac_score)
            
            # Prepare data for TDS metric
            try:
                # Use scores WITHOUT mask for TDS (to test true temporal memory)
                if len(z_scores_no_mask_list) == 0:
                    self._log_and_print('=' * 50)
                    self._log_and_print('TDS metric: No scores without mask available')
                    raise ValueError("No unmasked scores for TDS computation")
                
                scores_array = prepare_scores_for_metrics(z_scores_no_mask_list, timing_info_list)
                
                # Ensure connection_history has correct shape
                if len(connection_history_list.shape) == 4:
                    graphs_array = connection_history_list
                else:
                    n_samples = len(z_scores_list)
                    max_timesteps = scores_array.shape[1]
                    n_nodes = scores_array.shape[2]
                    graphs_array = np.zeros((n_samples, max_timesteps, n_nodes, n_nodes))
                    for i in range(n_samples):
                        min_t = min(max_timesteps, connection_history_list.shape[1] if len(connection_history_list.shape) > 1 else 1)
                        if len(connection_history_list.shape) == 3:
                            graphs_array[i, :min_t, :, :] = connection_history_list[i, :min_t, :, :]
                        else:
                            graphs_array[i, :min_t, :, :] = connection_history_list[:min_t, :, :]
                
                # Compute TDS metric
                self._log_and_print('=' * 50)
                self._log_and_print('Temporal Decoupling Score (TDS):')
                self._log_and_print('(Computed WITHOUT adjacency mask to test true temporal memory)')
                
                try:
                    min_timesteps_tds = min(scores_array.shape[1], graphs_array.shape[1])
                    scores_for_tds = scores_array[:, :min_timesteps_tds, :]
                    graphs_for_tds = graphs_array[:, :min_timesteps_tds, :, :]
                    
                    labels_array = np.array([label[self.window_size * 2:] for label in labels])
                    
                    if labels_array.ndim == 3:
                        labels_array = (labels_array.sum(axis=1) > 0).astype(float)
                    elif labels_array.ndim == 2:
                        if labels_array.shape[1] == scores_for_tds.shape[2]:
                            pass
                        else:
                            raise ValueError("Cannot convert timestep labels to node labels")
                    
                    tds_results = compute_temporal_decoupling_score(
                        scores_for_tds, graphs_for_tds, labels_array, timing_info_list
                    )
                    
                    tds_score = tds_results.get('tds_score', 0.0)
                    n_victims = tds_results.get('n_valid_victims', 0)
                    
                    self._log_and_print('  TDS Score: {:.5f} (based on {} victim nodes)', tds_score, n_victims)
                    self._log_and_print('  Interpretation: {}', interpret_tds(tds_score))
                    
                    if 'warning' in tds_results:
                        self._log_and_print('  Warning: {}', tds_results['warning'])
                    
                except Exception as e:
                    self._log_and_print('  TDS computation failed: {}', str(e))
                    import traceback
                    self._log_and_print('  Traceback: {}', traceback.format_exc())
                
                self._log_and_print('=' * 50)
                
            except Exception as e:
                self._log_and_print('Error computing new dynamic graph metrics: {}', str(e))

    def _testing_causal_discover(self, xs, causal_struct_value, connection_history_list=None, mode='static'):
        """
        Test causal discovery performance.
        
        Args:
            xs: List of time series data
            causal_struct_value: Static causal structure (for static inference)
            connection_history_list: List of dynamic adjacency matrices (for dynamic inference)
            mode: 'static' (default) or 'dynamic'
        """
        # Load model using ModelManager
        model_state = self.model_manager.load_model_state(self.model_dir, self.device)
        self.load_state_dict(model_state)
        self.eval()
        
        # Reset memories for testing
        self.encoder.reset_memory()
        
        # If mode is dynamic and connection history is available, use dynamic evaluation
        if mode == 'dynamic' and connection_history_list is not None:
            self._log_and_print('\n' + '='*80)
            self._log_and_print('DYNAMIC GRAPH CAUSAL DISCOVERY EVALUATION (StaR)')
            self._log_and_print('='*80)
            self._testing_dynamic_causal_discover(xs, connection_history_list)
            return
        
        # Otherwise, use static evaluation (original implementation)
        self._log_and_print('\n' + '='*80)
        self._log_and_print('STATIC GRAPH CAUSAL DISCOVERY EVALUATION (StaR)')
        self._log_and_print('='*80)
        
        encoder_causal_list = []
        with torch.no_grad():
            for x in xs:
                _, _, _, encoder_coeffs, _, _, _, _ = self._testing_step(x)
                encoder_estimate = torch.max(torch.median(torch.abs(encoder_coeffs), dim=0)[0],
                                             dim=0).values.cpu().numpy()
                encoder_causal_list.append(encoder_estimate)
        encoder_causal_struct_estimate_lst = np.stack(encoder_causal_list, axis=0)

        encoder_auroc = []
        encoder_auprc = []
        encoder_hamming = []
        encoder_f1 = []
        for i in range(len(encoder_causal_struct_estimate_lst)):
            encoder_auroc_temp, encoder_auprc_temp = eval_causal_structure(
                a_true=causal_struct_value, a_pred=encoder_causal_struct_estimate_lst[i])
            encoder_auroc.append(encoder_auroc_temp)
            encoder_auprc.append(encoder_auprc_temp)
            encoder_q = np.quantile(encoder_causal_struct_estimate_lst[i], q=self.causal_quantile)
            encoder_a_hat_binary = (encoder_causal_struct_estimate_lst[i] >= encoder_q).astype(float)
            _, _, _, _, ham_e = eval_causal_structure_binary(a_true=causal_struct_value,
                                                             a_pred=encoder_a_hat_binary)
            encoder_hamming.append(ham_e)
            encoder_f1.append(f1_score(causal_struct_value.flatten(), encoder_a_hat_binary.flatten()))
        self._log_and_print('Causal discovery F1: {:.5f} std: {:.5f}',
                            np.mean(encoder_f1), np.std(encoder_f1))
        self._log_and_print('Causal discovery AUROC: {:.5f} std: {:.5f}',
                            np.mean(encoder_auroc), np.std(encoder_auroc))
        self._log_and_print('Causal discovery AUPRC: {:.5f} std: {:.5f}',
                            np.mean(encoder_auprc), np.std(encoder_auprc))
        self._log_and_print('Causal discovery Hamming Distance: {:.5f} std: {:.5f}',
                            np.mean(encoder_hamming), np.std(encoder_hamming))
    
    def _testing_dynamic_causal_discover(self, xs, connection_history_list):
        """
        Test dynamic causal discovery performance using time-varying ground truth.
        
        Args:
            xs: List of time series data, shape (n_samples, T, num_vars)
            connection_history_list: List of dynamic adjacency matrices, 
                                    shape (n_samples, T-1, num_vars, num_vars)
        """
        from utils.dynamic_graph_metrics import evaluate_dynamic_causal_discovery, format_dynamic_metrics_report
        import time
        
        self._log_and_print('Testing dynamic causal discovery with TGN temporal memory...')
        self._log_and_print(f'Number of samples: {len(xs)}')
        
        all_results = []
        start_time = time.time()
        
        with torch.no_grad():
            for sample_idx, (x, connection_history) in enumerate(zip(xs, connection_history_list)):
                if sample_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    if sample_idx > 0:
                        avg_time = elapsed / sample_idx
                        remaining = avg_time * (len(xs) - sample_idx)
                        self._log_and_print(f'  Processing sample {sample_idx}/{len(xs)} '
                                          f'(elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s)...')
                    else:
                        self._log_and_print(f'  Processing sample {sample_idx}/{len(xs)}...')
                
                # Reset memory for each sample
                self.encoder.reset_memory()
                
                _, _, _, encoder_coeffs, _, _, _, _ = self._testing_step(x)
                
                num_windows = encoder_coeffs.shape[0]
                pred_graphs_per_timestep = []
                
                for window_idx in range(num_windows):
                    window_pred = torch.median(torch.abs(encoder_coeffs[window_idx]), dim=0)[0]
                    pred_graphs_per_timestep.append(window_pred.cpu().numpy())
                
                min_len = min(len(pred_graphs_per_timestep), len(connection_history))
                
                if min_len == 0:
                    self._log_and_print(f'Warning: Sample {sample_idx} has no overlapping timesteps, skipping...')
                    continue
                
                true_graphs = [connection_history[t] for t in range(min_len)]
                pred_graphs = pred_graphs_per_timestep[:min_len]
                
                try:
                    results = evaluate_dynamic_causal_discovery(
                        true_graphs=true_graphs,
                        pred_graphs=pred_graphs,
                        quantile=self.causal_quantile,
                        diagonal=False
                    )
                    all_results.append(results)
                except Exception as e:
                    self._log_and_print(f'Error evaluating sample {sample_idx}: {str(e)}')
                    continue
        
        if len(all_results) == 0:
            self._log_and_print('No valid results to aggregate!')
            return
        
        # Aggregate results across all samples
        aggregated_shd = []
        aggregated_fdr = []
        aggregated_tpr = []
        aggregated_mse = []
        aggregated_f1 = []
        
        for result in all_results:
            agg = result['aggregated']
            aggregated_shd.append(agg['SHD_mean'])
            aggregated_fdr.append(agg['FDR_mean'])
            aggregated_tpr.append(agg['TPR_mean'])
            aggregated_mse.append(agg['MSE_mean'])
            aggregated_f1.append(agg['F1_mean'])
        
        total_time = time.time() - start_time
        self._log_and_print('\n' + '='*80)
        self._log_and_print('DYNAMIC CAUSAL DISCOVERY RESULTS (StaR, Aggregated across all samples)')
        self._log_and_print('='*80)
        self._log_and_print('Structural Hamming Distance (SHD): {:.4f} +/- {:.4f}',
                           np.mean(aggregated_shd), np.std(aggregated_shd))
        self._log_and_print('False Discovery Rate (FDR):       {:.4f} +/- {:.4f}',
                           np.mean(aggregated_fdr), np.std(aggregated_fdr))
        self._log_and_print('True Positive Rate (TPR):          {:.4f} +/- {:.4f}',
                           np.mean(aggregated_tpr), np.std(aggregated_tpr))
        self._log_and_print('Mean Squared Error (MSE):          {:.4f} +/- {:.4f}',
                           np.mean(aggregated_mse), np.std(aggregated_mse))
        self._log_and_print('F1 Score:                          {:.4f} +/- {:.4f}',
                           np.mean(aggregated_f1), np.std(aggregated_f1))
        self._log_and_print('='*80)
        self._log_and_print('Note: Lower SHD, FDR, MSE are better; Higher TPR, F1 are better')
        self._log_and_print('TGN temporal memory should provide better tracking of dynamic causal structure')
        self._log_and_print(f'Total computation time: {total_time:.2f}s ({len(xs)} samples)\n')
