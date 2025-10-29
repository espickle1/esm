# %%
"""
Unified ESMC Protein Analysis Pipeline
Combines entropy calculation, logit coupling, and mutation importance scoring.

Usage:
    pipeline = ESMCProteinPipeline(device='cuda')
    results = pipeline.run(protein_logits_list)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: ENTROPY CALCULATION
# ============================================================================

class EntropyCalculator:
    """Calculate Shannon entropy from protein logits."""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def calculate_shannon_entropy(self, logits, base='e'):
        """
        Calculate Shannon entropy for each residue from logits.
        
        Args:
            logits: Tensor of shape (num_residues, num_tokens)
            base: 'e' (nats), '2' (bits), or '10' (dits)
        
        Returns:
            entropy: Tensor of shape (num_residues,)
        """
        logits = logits.to(self.device).float()
        probs = torch.softmax(logits, dim=-1)
        
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Convert base if needed
        if base == '2':
            entropy = entropy / np.log(2)
        elif base == '10':
            entropy = entropy / np.log(10)
        
        return entropy
    
    def calculate_entropy_batched(self, logits, base='e', batch_size=10000):
        """
        Calculate entropy for very large tensors using batching.
        
        Args:
            logits: Tensor of shape (num_residues, num_tokens)
            base: 'e' (nats), '2' (bits), or '10' (dits)
            batch_size: Number of residues per batch
        
        Returns:
            entropy: Tensor of shape (num_residues,)
        """
        logits = logits.to(self.device).float()
        num_residues = logits.shape[0]
        entropy_list = []
        
        for i in range(0, num_residues, batch_size):
            batch_end = min(i + batch_size, num_residues)
            logits_batch = logits[i:batch_end]
            
            probs_batch = torch.softmax(logits_batch, dim=-1)
            epsilon = 1e-10
            log_probs_batch = torch.log(probs_batch + epsilon)
            entropy_batch = -torch.sum(probs_batch * log_probs_batch, dim=1)
            
            # Handle NaN/Inf values (e.g., from all-zero logits producing uniform distributions)
            max_entropy = np.log(logits_batch.shape[1])  # log(vocab_size)
            entropy_batch = torch.where(
                torch.isnan(entropy_batch) | torch.isinf(entropy_batch),
                torch.ones_like(entropy_batch) * max_entropy,
                entropy_batch
            )
            
            entropy_list.append(entropy_batch)
        
        entropy = torch.cat(entropy_list, dim=0)
        
        if base == '2':
            entropy = entropy / np.log(2)
        elif base == '10':
            entropy = entropy / np.log(10)
        
        return entropy
    
    def get_constrained_flexible_positions(self, entropy):
        """
        Identify constrained (low entropy) and flexible (high entropy) positions.
        
        Returns:
            constrained: Indices of low entropy positions (bottom 10%)
            flexible: Indices of high entropy positions (top 10%)
        """
        entropy_float = entropy.float()
        p10 = torch.quantile(entropy_float, 0.1)
        p90 = torch.quantile(entropy_float, 0.9)
        
        constrained = torch.where(entropy_float < p10)[0]
        flexible = torch.where(entropy_float > p90)[0]
        
        return constrained, flexible


# ============================================================================
# PART 2: LOGIT COUPLING CALCULATION
# ============================================================================

class LogitCouplingCalculator:
    """Compute evolutionary coupling from ESMC logits."""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_coupling_from_logits(
        self, 
        logits: torch.Tensor,
        method: str = "frobenius"
    ) -> np.ndarray:
        """
        Compute coupling matrix directly from logits.
        
        Args:
            logits: Tensor of shape (seq_len, vocab_size)
            method: 'frobenius' (distribution distance) | 'covariance' | 'mutual_information'
        
        Returns:
            Coupling matrix (seq_len, seq_len)
        """
        logits = logits.to(self.device).float()
        
        if method == "frobenius":
            return self._frobenius_coupling(logits)
        elif method == "covariance":
            return self._covariance_coupling(logits)
        elif method == "mutual_information":
            return self._mi_coupling(logits)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _frobenius_coupling(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute coupling based on Frobenius norm of outer product.
        Measures similarity of predicted distributions at each position.
        """
        probs = torch.softmax(logits, dim=-1)
        coupling_gpu = torch.mm(probs, probs.t())
        coupling = coupling_gpu.cpu().numpy()
        coupling = (coupling + coupling.T) / 2
        return coupling
    
    def _covariance_coupling(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute coupling based on covariance of logit vectors.
        High covariance suggests positions co-vary in the learned space.
        Robust to zero vectors.
        """
        logits = logits.float()
        logits_centered = logits - logits.mean(dim=1, keepdim=True)
        
        # Use larger epsilon for numerical stability and add small noise to avoid zero vectors
        logits_norm = torch.norm(logits_centered, dim=1, keepdim=True) + 1e-6
        logits_normalized = logits_centered / logits_norm
        
        correlation_matrix = torch.mm(logits_normalized, logits_normalized.t())
        coupling_gpu = torch.abs(correlation_matrix)
        
        # Handle NaN/Inf values
        coupling_gpu = torch.where(
            torch.isnan(coupling_gpu) | torch.isinf(coupling_gpu),
            torch.zeros_like(coupling_gpu),
            coupling_gpu
        )
        
        coupling = coupling_gpu.cpu().numpy()
        coupling = (coupling + coupling.T) / 2
        
        # Ensure no NaN in output
        coupling = np.nan_to_num(coupling, nan=0.0, posinf=0.0, neginf=0.0)
        
        return coupling
    
    def _mi_coupling(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute coupling based on mutual information between predictions.
        """
        probs = torch.softmax(logits, dim=-1).float()
        seq_len = probs.shape[0]
        
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        log_probs_expanded_i = log_probs.unsqueeze(1)
        log_probs_expanded_j = log_probs.unsqueeze(0)
        probs_expanded_i = probs.unsqueeze(1)
        
        kl_matrix = torch.sum(
            probs_expanded_i * (log_probs_expanded_i - log_probs_expanded_j),
            dim=2
        )
        
        entropy_i = entropy.unsqueeze(1)
        entropy_j = entropy.unsqueeze(0)
        
        mi_matrix = entropy_i + entropy_j - torch.abs(kl_matrix)
        mi_matrix = mi_matrix / (mi_matrix.max() + 1e-8)
        mi_matrix = torch.clamp(mi_matrix, min=0)
        
        coupling = mi_matrix.cpu().numpy()
        return coupling
    
    def apc_correction(self, coupling: np.ndarray) -> np.ndarray:
        """
        Apply Average Product Correction to coupling matrix.
        Removes background correlations.
        Robust to all-zero or near-zero coupling matrices.
        """
        seq_len = coupling.shape[0]
        
        # Compute row and column averages
        row_avgs = coupling.mean(axis=1)
        col_avgs = coupling.mean(axis=0)
        global_avg = coupling.mean()
        
        # Protect against division by zero
        if np.abs(global_avg) < 1e-12:
            # If global average is near zero, return original coupling
            return coupling
        
        # Apply APC
        coupling_apc = np.zeros_like(coupling)
        for i in range(seq_len):
            for j in range(seq_len):
                coupling_apc[i, j] = coupling[i, j] - (row_avgs[i] * col_avgs[j] / global_avg)
        
        # Ensure no NaN/Inf in output
        coupling_apc = np.nan_to_num(coupling_apc, nan=0.0, posinf=0.0, neginf=0.0)
        
        return coupling_apc
    
    def compute_direct_information(self, coupling: np.ndarray) -> np.ndarray:
        """
        Compute Direct Information from coupling matrix.
        Better for identifying true coevolution.
        Robust to zero coupling matrices.
        """
        seq_len = coupling.shape[0]
        di = np.zeros_like(coupling)
        
        # Use row sums as proxy for direct information
        for i in range(seq_len):
            row_sum = coupling[i, :].sum()
            if row_sum > 1e-12:  # Only normalize if sum is non-trivial
                di[i, :] = coupling[i, :] / (row_sum + 1e-10)
            else:
                di[i, :] = 0.0
        
        # Ensure no NaN/Inf in output
        di = np.nan_to_num(di, nan=0.0, posinf=0.0, neginf=0.0)
        
        return di


# ============================================================================
# PART 3: MUTATION IMPORTANCE SCORING
# ============================================================================

class MutationImportanceScorer:
    """Score mutation importance based on entropy and coupling."""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def score_mutation_importance(
        self, 
        position: int,
        entropy: torch.Tensor,
        coupling_matrix: torch.Tensor,
        other_mutations: Optional[List[int]] = None
    ) -> Dict:
        """
        Score importance of a mutation at a given position.
        
        Args:
            position: Position of mutation
            entropy: Entropy tensor
            coupling_matrix: Coupling matrix
            other_mutations: Positions of other mutations (if None, use all positions)
        
        Returns:
            Dict with constraint, coupling, and importance scores
        """
        entropy = entropy.to(self.device).float()
        coupling_matrix = coupling_matrix.to(self.device).float()
        
        if other_mutations is None:
            other_mutations = list(range(len(entropy)))
        
        if isinstance(other_mutations, list):
            other_mutations = torch.tensor(other_mutations, device=self.device)
        else:
            other_mutations = other_mutations.to(self.device)
        
        # Compute scores
        constraint = 1.0 / (entropy[position] + 0.1)
        coupling_to_adaptive = coupling_matrix[position, other_mutations].mean()
        overall_coupling = coupling_matrix[position, :].mean()
        
        importance = constraint * coupling_to_adaptive
        independence = constraint / (overall_coupling + 0.1)
        
        return {
            'position': position,
            'constraint': constraint.item(),
            'coupling_to_adaptive': coupling_to_adaptive.item(),
            'overall_coupling': overall_coupling.item(),
            'importance': importance.item(),
            'independence': independence.item()
        }
    
    def score_all_mutations(
        self,
        entropy: torch.Tensor,
        coupling_matrix: torch.Tensor,
        threshold_entropy: Optional[float] = None,
        threshold_coupling: Optional[float] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Vectorized scoring of all mutations.
        
        Args:
            entropy: Entropy tensor (num_residues,)
            coupling_matrix: Coupling matrix (num_residues, num_residues)
            threshold_entropy: Only score positions with entropy below this
            threshold_coupling: Only score positions with coupling above this
        
        Returns:
            Dict with all scores for all positions
        """
        entropy = entropy.to(device).float()
        coupling_matrix = coupling_matrix.to(device).float()
        
        num_residues = entropy.shape[0]
        
        # Filter positions
        if threshold_entropy is not None:
            mask = entropy < threshold_entropy
            positions = torch.where(mask)[0]
        elif threshold_coupling is not None:
            overall_coupling = coupling_matrix.mean(dim=1)
            mask = overall_coupling > threshold_coupling
            positions = torch.where(mask)[0]
        else:
            positions = torch.arange(num_residues, device=device)
        
        # Vectorized computation
        constraint = 1.0 / (entropy + 0.1)
        overall_coupling = coupling_matrix.mean(dim=1)
        importance = constraint * overall_coupling
        independence = constraint / (overall_coupling + 0.1)
        
        return {
            'positions': positions.cpu().numpy(),
            'constraint': constraint.cpu().numpy(),
            'coupling_to_adaptive': overall_coupling.cpu().numpy(),
            'overall_coupling': overall_coupling.cpu().numpy(),
            'importance': importance.cpu().numpy(),
            'independence': independence.cpu().numpy(),
            'num_scored': len(positions)
        }
    
    def classify_mutations(
        self,
        entropy: torch.Tensor,
        coupling_matrix: torch.Tensor,
        entropy_percentile: float = 0.3,
        coupling_percentile: float = 0.7
    ) -> Dict:
        """
        Classify mutations into 4 types based on entropy and coupling.
        Robust to edge cases like all identical values.
        
        Type 1: Low entropy, low coupling -> Critical core
        Type 2: High entropy, low coupling -> Surface tuning
        Type 3: Low entropy, high coupling -> Structural hub
        Type 4: High entropy, high coupling -> Flexible interface
        """
        entropy = entropy.to(self.device).float()
        coupling_matrix = coupling_matrix.to(self.device).float()
        
        # Compute thresholds with edge case handling
        entropy_std = entropy.std().item()
        overall_coupling = coupling_matrix.mean(dim=1)
        coupling_std = overall_coupling.std().item()
        
        # If all values are identical (std ≈ 0), skip classification
        if entropy_std < 1e-10 and coupling_std < 1e-10:
            # All positions are identical - assign all to type 1
            num_residues = len(entropy)
            return {
                'type1_critical_core': np.arange(num_residues),
                'type2_surface_tuning': np.array([]),
                'type3_structural_hub': np.array([]),
                'type4_flexible_interface': np.array([]),
                'entropy_threshold': entropy.mean().item(),
                'coupling_threshold': overall_coupling.mean().item(),
                'note': 'All values identical - classification not meaningful'
            }
        
        # Compute thresholds
        entropy_threshold = torch.quantile(entropy, entropy_percentile)
        coupling_threshold = torch.quantile(overall_coupling, coupling_percentile)
        
        # Classify
        low_entropy = entropy < entropy_threshold
        high_entropy = entropy >= entropy_threshold
        low_coupling = overall_coupling < coupling_threshold
        high_coupling = overall_coupling >= coupling_threshold
        
        type1 = torch.where(low_entropy & low_coupling)[0]  # Critical core
        type2 = torch.where(high_entropy & low_coupling)[0]  # Surface tuning
        type3 = torch.where(low_entropy & high_coupling)[0]  # Structural hub
        type4 = torch.where(high_entropy & high_coupling)[0]  # Flexible interface
        
        return {
            'type1_critical_core': type1.cpu().numpy(),
            'type2_surface_tuning': type2.cpu().numpy(),
            'type3_structural_hub': type3.cpu().numpy(),
            'type4_flexible_interface': type4.cpu().numpy(),
            'entropy_threshold': entropy_threshold.item(),
            'coupling_threshold': coupling_threshold.item()
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ESMCProteinPipeline:
    """
    Unified pipeline for ESMC protein analysis.
    Combines entropy, coupling, and mutation importance scoring.
    """
    
    def __init__(self, device='cuda', coupling_method='frobenius'):
        """
        Initialize the pipeline.
        
        Args:
            device: 'cuda' or 'cpu'
            coupling_method: 'frobenius', 'covariance', or 'mutual_information'
        """
        self.device = device
        self.coupling_method = coupling_method
        
        self.entropy_calc = EntropyCalculator(device=device)
        self.coupling_calc = LogitCouplingCalculator(device=device)
        self.scorer = MutationImportanceScorer(device=device)
    
    def process_single_protein(
        self,
        logits: torch.Tensor,
        protein_id: str = None,
        apply_apc: bool = True,
        compute_direct_info: bool = False,
        batch_size: int = 10000
    ) -> Dict:
        """
        Process a single protein through the complete pipeline.
        Robust to all-zero logits and other edge cases.
        
        Args:
            logits: Protein logits (num_residues, num_tokens)
            protein_id: Optional identifier for protein
            apply_apc: Whether to apply APC correction to coupling
            compute_direct_info: Whether to compute direct information
            batch_size: Batch size for entropy calculation
        
        Returns:
            Dict with all results for this protein
        """
        # Validate input - check for None
        if logits is None:
            raise ValueError(f"Protein {protein_id or 'unknown'} has None/null logits!")
        
        # Move to device
        logits = logits.to(self.device).float()
        
        # Validate input and detect problematic logits
        num_zero_rows = (logits.abs().sum(dim=1) < 1e-10).sum().item()
        if num_zero_rows > 0:
            print(f"  ⚠️  Warning: {num_zero_rows} residues with all-zero logits - will be handled gracefully")
        
        # 1. Calculate entropy
        print(f"  Computing entropy for protein {protein_id or '...'}")
        entropy = self.entropy_calc.calculate_entropy_batched(
            logits, base='e', batch_size=batch_size
        )
        
        # Check for NaN/Inf in entropy and replace with max entropy
        entropy = torch.where(
            torch.isnan(entropy) | torch.isinf(entropy),
            torch.ones_like(entropy) * np.log(logits.shape[1]),
            entropy
        )
        
        constrained, flexible = self.entropy_calc.get_constrained_flexible_positions(entropy)
        
        # 2. Calculate coupling
        print(f"  Computing coupling matrix (method: {self.coupling_method})")
        coupling = self.coupling_calc.compute_coupling_from_logits(
            logits, method=self.coupling_method
        )
        
        # Validate coupling matrix
        coupling = np.nan_to_num(coupling, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply APC correction if requested
        coupling_apc = None
        if apply_apc:
            print(f"  Applying APC correction")
            coupling_apc = self.coupling_calc.apc_correction(coupling)
        
        # Compute direct information if requested
        direct_info = None
        if compute_direct_info:
            print(f"  Computing direct information")
            direct_info = self.coupling_calc.compute_direct_information(coupling)
        
        # Convert coupling to tensor for scoring
        coupling_tensor = torch.from_numpy(coupling_apc if coupling_apc is not None else coupling).to(self.device).float()
        
        # 3. Score mutation importance
        print(f"  Scoring mutation importance")
        mutation_scores = self.scorer.score_all_mutations(
            entropy, coupling_tensor, device=self.device
        )
        
        # Classify mutations
        mutation_types = self.scorer.classify_mutations(entropy, coupling_tensor)
        
        # Return results
        results = {
            'protein_id': protein_id,
            'num_residues': logits.shape[0],
            'num_zero_logit_residues': num_zero_rows,
            'entropy': entropy.cpu().numpy(),
            'constrained_positions': constrained.cpu().numpy(),
            'flexible_positions': flexible.cpu().numpy(),
            'coupling': coupling,
            'coupling_apc': coupling_apc,
            'direct_information': direct_info,
            'mutation_scores': mutation_scores,
            'mutation_types': mutation_types,
            'statistics': {
                'entropy_mean': entropy.mean().item(),
                'entropy_std': entropy.std().item(),
                'entropy_min': entropy.min().item(),
                'entropy_max': entropy.max().item(),
                'coupling_mean': coupling.mean(),
                'coupling_max': coupling.max(),
            }
        }
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return results
    
    def run(
        self,
        protein_logits_list: List[torch.Tensor],
        protein_ids: Optional[List[str]] = None,
        apply_apc: bool = True,
        compute_direct_info: bool = False,
        batch_size: int = 10000
    ) -> List[Dict]:
        """
        Process multiple proteins through the complete pipeline.
        Handles None/null entries gracefully.
        
        Args:
            protein_logits_list: List of logit tensors (one per protein)
            protein_ids: Optional list of protein identifiers
            apply_apc: Whether to apply APC correction
            compute_direct_info: Whether to compute direct information
            batch_size: Batch size for entropy calculation
        
        Returns:
            List of result dicts (one per protein)
        """
        # Filter out None entries and track them
        valid_indices = []
        skipped_indices = []
        
        for i, logits in enumerate(protein_logits_list):
            if logits is None:
                skipped_indices.append(i)
            else:
                valid_indices.append(i)
        
        # Report skipped entries
        if skipped_indices:
            print(f"\n⚠️  WARNING: {len(skipped_indices)} proteins have None/null logits and will be skipped:")
            for idx in skipped_indices[:10]:  # Show first 10
                pid = protein_ids[idx] if protein_ids else f"protein_{idx}"
                print(f"     - {pid} (index {idx})")
            if len(skipped_indices) > 10:
                print(f"     ... and {len(skipped_indices) - 10} more")
            print()
        
        # Filter to valid entries
        filtered_logits = [protein_logits_list[i] for i in valid_indices]
        
        if protein_ids is None:
            filtered_ids = [f"protein_{i}" for i in valid_indices]
        else:
            filtered_ids = [protein_ids[i] for i in valid_indices]
        
        # Check if all entries were None
        if not filtered_logits:
            print("❌ ERROR: All protein logits are None/null!")
            return []
        
        num_proteins = len(filtered_logits)
        
        print(f"\n{'='*70}")
        print(f"ESMC Protein Analysis Pipeline")
        print(f"{'='*70}")
        print(f"Processing {num_proteins} proteins (skipped {len(skipped_indices)} None entries)")
        print(f"Device: {self.device}")
        print(f"Coupling method: {self.coupling_method}")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for i, (logits, protein_id) in enumerate(zip(filtered_logits, filtered_ids)):
            print(f"\n[{i+1}/{num_proteins}] Processing {protein_id}")
            print(f"  Protein size: {logits.shape[0]} residues × {logits.shape[1]} tokens")
            
            try:
                results = self.process_single_protein(
                    logits,
                    protein_id=protein_id,
                    apply_apc=apply_apc,
                    compute_direct_info=compute_direct_info,
                    batch_size=batch_size
                )
                all_results.append(results)
            except Exception as e:
                print(f"  ❌ ERROR processing {protein_id}: {str(e)}")
                print(f"     Skipping this protein")
                continue
        
        print(f"\n{'='*70}")
        print(f"Pipeline complete! Processed {len(all_results)} proteins")
        if skipped_indices:
            print(f"Skipped {len(skipped_indices)} proteins (None/null entries)")
        print(f"{'='*70}\n")
        
        return all_results
    
    def get_summary_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute summary statistics across all proteins.
        
        Args:
            results: List of result dicts from run()
        
        Returns:
            Dict with aggregate statistics
        """
        entropies = [r['entropy'] for r in results]
        all_entropy = np.concatenate(entropies)
        
        num_residues_total = sum(r['num_residues'] for r in results)
        num_proteins = len(results)
        
        summary = {
            'num_proteins': num_proteins,
            'num_residues_total': num_residues_total,
            'avg_residues_per_protein': num_residues_total / num_proteins,
            'global_entropy_mean': all_entropy.mean(),
            'global_entropy_std': all_entropy.std(),
            'global_entropy_min': all_entropy.min(),
            'global_entropy_max': all_entropy.max(),
            'per_protein_stats': [r['statistics'] for r in results]
        }
        
        return summary


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_protein_logits_from_dict(protein_dict_list: List[Dict], device='cuda') -> List[torch.Tensor]:
    """
    Extract logits from a list of protein dictionaries.
    
    Args:
        protein_dict_list: List of dicts with 'logits' key
        device: Device to move tensors to
    
    Returns:
        List of logit tensors
    """
    logits_list = []
    
    for i, protein_dict in enumerate(protein_dict_list):
        if 'logits' in protein_dict:
            logits = protein_dict['logits']
            
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits).float()
            else:
                logits = logits.float()
            
            logits = logits.to(device)
            logits_list.append(logits)
        else:
            print(f"Warning: Protein {i} missing 'logits' key, skipping...")
    
    return logits_list


def create_results_summary_table(results: List[Dict]) -> dict:
    """
    Create a summary table of results for all proteins.
    
    Args:
        results: List of result dicts from pipeline
    
    Returns:
        Dict suitable for conversion to DataFrame or other formats
    """
    summary_data = []
    
    for result in results:
        stats = result['statistics']
        summary_data.append({
            'protein_id': result['protein_id'],
            'num_residues': result['num_residues'],
            'entropy_mean': stats['entropy_mean'],
            'entropy_std': stats['entropy_std'],
            'coupling_mean': stats['coupling_mean'],
            'coupling_max': stats['coupling_max'],
            'num_constrained': len(result['constrained_positions']),
            'num_flexible': len(result['flexible_positions']),
            'num_type1_critical': len(result['mutation_types']['type1_critical_core']),
            'num_type2_surface': len(result['mutation_types']['type2_surface_tuning']),
            'num_type3_hub': len(result['mutation_types']['type3_structural_hub']),
            'num_type4_flexible': len(result['mutation_types']['type4_flexible_interface']),
        })
    
    return summary_data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the pipeline with mock data.
    """
    
    print("\n" + "="*70)
    print("ESMC Protein Analysis Pipeline - Example Usage")
    print("="*70 + "\n")
    
    # Create mock protein logits
    num_proteins = 3
    protein_logits_list = []
    protein_ids = []
    
    for i in range(num_proteins):
        seq_len = np.random.randint(1000, 3000)
        vocab_size = 20
        logits = torch.randn(seq_len, vocab_size)
        
        # Add some structure
        for j in range(0, seq_len, 100):
            for k in range(j+10, min(j+60, seq_len), 20):
                logits[k] = logits[j] + torch.randn(vocab_size) * 0.3
        
        protein_logits_list.append(logits)
        protein_ids.append(f"protein_{i+1}")
    
    # Initialize pipeline
    pipeline = ESMCProteinPipeline(device='cuda', coupling_method='frobenius')
    
    # Run pipeline
    results = pipeline.run(
        protein_logits_list,
        protein_ids=protein_ids,
        apply_apc=True,
        compute_direct_info=True,
        batch_size=10000
    )
    
    # Get summary statistics
    summary = pipeline.get_summary_statistics(results)
    
    print("\nGlobal Statistics:")
    print(f"  Total proteins: {summary['num_proteins']}")
    print(f"  Total residues: {summary['num_residues_total']}")
    print(f"  Avg residues/protein: {summary['avg_residues_per_protein']:.0f}")
    print(f"  Global entropy: {summary['global_entropy_mean']:.4f} ± {summary['global_entropy_std']:.4f}")
    
    # Create summary table
    summary_table = create_results_summary_table(results)
    
    print("\nPer-Protein Summary:")
    for row in summary_table:
        print(f"\n  {row['protein_id']}:")
        print(f"    Residues: {row['num_residues']}")
        print(f"    Entropy: {row['entropy_mean']:.4f} ± {row['entropy_std']:.4f}")
        print(f"    Constrained positions: {row['num_constrained']}")
        print(f"    Flexible positions: {row['num_flexible']}")
        print(f"    Type 1 (Critical core): {row['num_type1_critical']}")
        print(f"    Type 2 (Surface tuning): {row['num_type2_surface']}")
        print(f"    Type 3 (Structural hub): {row['num_type3_hub']}")
        print(f"    Type 4 (Flexible interface): {row['num_type4_flexible']}")
