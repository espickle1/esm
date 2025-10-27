"""
ESMC Logit-Based Evolutionary Coupling Analysis

Computes evolutionary coupling between residues using the model's prediction logits.
This measures how mutations at one position affect predictions at another position.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm


class ESMCLogitCoupling:
    """
    Compute evolutionary coupling from ESMC logits.
    
    Two main approaches:
    1. Direct method: You already have logits, compute coupling from them
    2. Perturbation method: Mutate position i, see how logits at position j change
    """
    
    def __init__(self, model=None, tokenizer=None):
        """
        Initialize coupling analyzer.
        
        Args:
            model: ESMC model (only needed for perturbation method)
            tokenizer: ESMC tokenizer (only needed for perturbation method)
        """
        self.model = model
        self.tokenizer = tokenizer
        if model is not None:
            self.model.eval()
    
    def compute_coupling_from_logits(
        self, 
        logits: torch.Tensor,
        method: str = "frobenius"
    ) -> np.ndarray:
        """
        Compute coupling matrix directly from logits.
        
        This uses the covariance of logit vectors as a proxy for coupling.
        High covariance suggests positions are evolutionarily linked.
        
        Args:
            logits: Tensor of shape (seq_len, vocab_size) - ESMC output logits
            method: 'frobenius' (distance between distributions) or 
                    'covariance' (covariance of logits) or
                    'mutual_information' (MI between predicted distributions)
            
        Returns:
            Coupling matrix (seq_len, seq_len)
        """
        seq_len, vocab_size = logits.shape
        
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
        
        Measures how similar the predicted distributions are at each position.
        High similarity suggests coordinated evolution.
        GPU-accelerated vectorized version.
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)  # (seq_len, vocab_size)
        
        # Vectorized: compute all pairwise dot products at once
        # Using matrix multiplication: (seq_len, vocab_size) @ (vocab_size, seq_len) = (seq_len, seq_len)
        coupling_gpu = torch.mm(probs, probs.t())  # Stay on GPU
        
        # Convert to numpy for return
        coupling = coupling_gpu.cpu().numpy()
        
        # Ensure symmetry
        coupling = (coupling + coupling.T) / 2
        
        return coupling
    
    def _covariance_coupling(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute coupling based on covariance of logit vectors.
        
        High covariance suggests positions co-vary in the learned space.
        GPU-accelerated vectorized version.
        """
        # Ensure tensor is on same device
        logits = logits.float()
        
        # Center the logits on GPU
        logits_centered = logits - logits.mean(dim=1, keepdim=True)
        
        # Compute correlation matrix using vectorized operations
        # Normalize each row
        logits_norm = torch.norm(logits_centered, dim=1, keepdim=True) + 1e-8
        logits_normalized = logits_centered / logits_norm
        
        # Compute correlation matrix: all pairwise dot products (stays on GPU)
        correlation_matrix = torch.mm(logits_normalized, logits_normalized.t())
        
        # Take absolute value for coupling
        coupling_gpu = torch.abs(correlation_matrix)
        
        # Convert to numpy
        coupling = coupling_gpu.cpu().numpy()
        
        # Ensure symmetry
        coupling = (coupling + coupling.T) / 2
        
        return coupling
    
    def _mi_coupling(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute coupling based on mutual information between predictions.
        
        Estimates MI from the predicted probability distributions.
        GPU-accelerated vectorized version.
        """
        probs = torch.softmax(logits, dim=-1).float()  # Keep on GPU
        seq_len = probs.shape[0]
        
        # Compute entropy for all positions at once
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * log_probs, dim=1)  # (seq_len,)
        
        # Compute pairwise KL divergence matrix (vectorized)
        # KL(P||Q) = sum(P * log(P/Q))
        log_probs_expanded_i = log_probs.unsqueeze(1)  # (seq_len, 1, vocab_size)
        log_probs_expanded_j = log_probs.unsqueeze(0)  # (1, seq_len, vocab_size)
        probs_expanded_i = probs.unsqueeze(1)  # (seq_len, 1, vocab_size)
        
        # Compute KL divergence for all pairs
        kl_matrix = torch.sum(
            probs_expanded_i * (log_probs_expanded_i - log_probs_expanded_j),
            dim=2
        )  # (seq_len, seq_len)
        
        # Compute mutual information proxy
        entropy_i = entropy.unsqueeze(1)  # (seq_len, 1)
        entropy_j = entropy.unsqueeze(0)  # (1, seq_len)
        
        mi_matrix = entropy_i + entropy_j - torch.abs(kl_matrix)
        
        # Normalize
        mi_matrix = mi_matrix / (mi_matrix.max() + 1e-8)
        mi_matrix = torch.clamp(mi_matrix, min=0)  # Keep only positive values
        
        # Convert to numpy
        coupling = mi_matrix.cpu().numpy()
        
        return coupling
    
    def compute_perturbation_coupling(
        self,
        sequence: str,
        positions: Optional[List[int]] = None,
        mutations: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute coupling by perturbation: mutate position i, measure effect on j.
        
        This is more computationally expensive but more direct.
        It measures: "If I mutate residue i, how much do predictions at j change?"
        
        Args:
            sequence: Original protein sequence
            positions: Which positions to perturb (None = all)
            mutations: Which amino acids to mutate to (None = all 20 standard AA)
            
        Returns:
            Coupling matrix (seq_len, seq_len)
        """
        if self.model is None:
            raise ValueError("Model required for perturbation method")
        
        seq_len = len(sequence)
        
        if positions is None:
            positions = list(range(seq_len))
        
        if mutations is None:
            mutations = list("ACDEFGHIKLMNPQRSTVWY")
        
        # Get wild-type logits
        wt_logits = self._get_logits(sequence)
        
        # Compute coupling matrix
        coupling = np.zeros((seq_len, seq_len))
        
        print(f"Computing perturbation coupling for {len(positions)} positions...")
        for pos_i in tqdm(positions):
            original_aa = sequence[pos_i]
            
            # Try each mutation
            mutation_effects = []
            
            for mut_aa in mutations:
                if mut_aa == original_aa:
                    continue
                
                # Create mutant sequence
                mut_seq = sequence[:pos_i] + mut_aa + sequence[pos_i+1:]
                
                # Get mutant logits
                mut_logits = self._get_logits(mut_seq)
                
                # Compute effect on all positions
                effect = torch.norm(mut_logits - wt_logits, dim=-1).cpu().numpy()
                mutation_effects.append(effect)
            
            # Average effect across all mutations at position i
            if mutation_effects:
                avg_effect = np.mean(mutation_effects, axis=0)
                coupling[pos_i, :] = avg_effect
        
        # Make symmetric
        coupling = (coupling + coupling.T) / 2
        
        return coupling
    
    def _get_logits(self, sequence: str) -> torch.Tensor:
        """Get logits for a sequence."""
        with torch.no_grad():
            if self.tokenizer:
                inputs = self.tokenizer(sequence, return_tensors="pt")
            else:
                inputs = {"input_ids": sequence}
            
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)  # (seq_len, vocab_size)
        
        return logits
    
    def compute_coupling_from_logits_gpu(
        self, 
        logits: torch.Tensor,
        method: str = "frobenius",
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Compute coupling matrix directly from logits, keeping result on GPU.
        
        Args:
            logits: Tensor of shape (seq_len, vocab_size)
            method: 'frobenius', 'covariance', or 'mutual_information'
            device: 'cuda' or 'cpu'
        
        Returns:
            Coupling matrix tensor on GPU
        """
        logits = logits.to(device)
        
        if method == "frobenius":
            # Vectorized frobenius directly on GPU
            probs = torch.softmax(logits, dim=-1)
            coupling_gpu = torch.mm(probs, probs.t())
            return coupling_gpu
        elif method == "covariance":
            # Vectorized covariance on GPU
            logits_centered = logits - logits.mean(dim=1, keepdim=True)
            logits_norm = torch.norm(logits_centered, dim=1, keepdim=True) + 1e-8
            logits_normalized = logits_centered / logits_norm
            coupling_gpu = torch.mm(logits_normalized, logits_normalized.t())
            return torch.abs(coupling_gpu)
        elif method == "mutual_information":
            # This returns a cpu tensor in the current implementation
            # For full GPU version, this would need scipy port
            return torch.from_numpy(self._mi_coupling(logits)).to(device)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def clear_gpu_cache(self):
        """Clear GPU cache."""
        torch.cuda.empty_cache()
    
    def compute_direct_information(
        self, 
        coupling: np.ndarray,
        threshold: Optional[float] = None,
        device: str = 'cuda'
    ) -> np.ndarray:
        """
        Compute Direct Information (DI) from coupling matrix.
        
        DI removes indirect correlations using matrix inversion.
        This is similar to what DCA does.
        GPU-accelerated version.
        
        Args:
            coupling: Raw coupling matrix
            threshold: Optional threshold to zero out weak couplings
            device: 'cuda' or 'cpu'
            
        Returns:
            Direct information matrix
        """
        # Convert to GPU tensor
        if isinstance(coupling, np.ndarray):
            coupling_gpu = torch.from_numpy(coupling).float().to(device)
        else:
            coupling_gpu = coupling.float().to(device)
        
        # Add small value to diagonal for numerical stability
        coupling_reg = coupling_gpu + torch.eye(coupling_gpu.shape[0], device=device) * 1e-4
        
        # Invert to get precision matrix (removes indirect effects)
        try:
            precision = torch.linalg.inv(coupling_reg)
        except RuntimeError:
            # Use pseudo-inverse if singular
            precision = torch.linalg.pinv(coupling_reg)
        
        # DI is negative precision (off-diagonal elements)
        di_gpu = -precision
        di_gpu.fill_diagonal_(0)
        
        # Take absolute value and normalize
        di_gpu = torch.abs(di_gpu)
        di_max = di_gpu.max()
        if di_max > 0:
            di_gpu = di_gpu / di_max
        
        if threshold:
            di_gpu[di_gpu < threshold] = 0
        
        # Convert back to numpy
        di = di_gpu.cpu().numpy()
        
        return di
    
    def apc_correction(self, coupling: np.ndarray) -> np.ndarray:
        """
        Apply Average Product Correction (APC).
        
        This removes phylogenetic and entropic biases.
        Standard in contact prediction.
        GPU-accelerated version.
        
        Args:
            coupling: Raw coupling matrix (numpy or torch tensor)
            
        Returns:
            APC-corrected coupling matrix
        """
        # Convert to GPU tensor if needed
        if isinstance(coupling, np.ndarray):
            coupling_gpu = torch.from_numpy(coupling).float().cuda()
        else:
            coupling_gpu = coupling.float().cuda()
        
        # Compute row and column means (all on GPU)
        row_means = coupling_gpu.mean(dim=1, keepdim=True)
        col_means = coupling_gpu.mean(dim=0, keepdim=True)
        total_mean = coupling_gpu.mean()
        
        # APC correction (vectorized on GPU)
        apc_matrix = (row_means @ col_means) / total_mean
        corrected_gpu = coupling_gpu - apc_matrix
        
        # Keep only positive values
        corrected_gpu = torch.clamp(corrected_gpu, min=0)
        
        # Convert back to numpy
        corrected = corrected_gpu.cpu().numpy()
        
        return corrected


def batch_compute_coupling_gpu(
    logits_list: List[torch.Tensor],
    method: str = "frobenius",
    device: str = 'cuda',
    batch_size: int = 10
) -> List[np.ndarray]:
    """
    Compute coupling matrices for multiple proteins efficiently on GPU.
    
    Args:
        logits_list: List of logit tensors
        method: Coupling method to use
        device: 'cuda' or 'cpu'
        batch_size: Number of proteins to process before clearing cache
    
    Returns:
        List of coupling matrices
    """
    analyzer = ESMCLogitCoupling()
    results = []
    
    for i, logits in enumerate(logits_list):
        coupling = analyzer.compute_coupling_from_logits(logits.to(device), method=method)
        results.append(coupling)
        
        # Clear cache periodically
        if (i + 1) % batch_size == 0:
            torch.cuda.empty_cache()
    
    return results


def compare_coupling_methods_gpu(
    logits: torch.Tensor,
    device: str = 'cuda',
    apply_apc: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Compare all coupling methods on GPU and return results.
    
    Args:
        logits: Logit tensor
        device: 'cuda' or 'cpu'
        apply_apc: Whether to apply APC correction
        save_path: Optional path to save comparison plot
    
    Returns:
        Dict of coupling matrices by method
    """
    analyzer = ESMCLogitCoupling()
    logits = logits.to(device)
    
    results = {}
    
    # Frobenius
    print("Computing Frobenius coupling...")
    coupling_frob = analyzer.compute_coupling_from_logits(logits, method="frobenius")
    results['frobenius'] = coupling_frob
    if apply_apc:
        results['frobenius_apc'] = analyzer.apc_correction(coupling_frob)
    
    # Covariance
    print("Computing Covariance coupling...")
    coupling_cov = analyzer.compute_coupling_from_logits(logits, method="covariance")
    results['covariance'] = coupling_cov
    if apply_apc:
        results['covariance_apc'] = analyzer.apc_correction(coupling_cov)
    
    # Mutual Information
    print("Computing MI coupling...")
    coupling_mi = analyzer.compute_coupling_from_logits(logits, method="mutual_information")
    results['mutual_information'] = coupling_mi
    
    # Direct Information
    print("Computing Direct Information...")
    results['direct_information'] = analyzer.compute_direct_information(
        results['frobenius_apc'] if apply_apc else coupling_frob,
        device=device
    )
    
    torch.cuda.empty_cache()
    
    return results


def visualize_coupling(
    coupling: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Evolutionary Coupling",
    top_k: int = 50,
    min_separation: int = 6,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize coupling matrix with multiple panels.
    
    Args:
        coupling: Coupling matrix
        sequence: Protein sequence (optional)
        title: Main title
        top_k: Number of top contacts to highlight
        min_separation: Minimum sequence separation
        save_path: Path to save figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Full coupling matrix
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(coupling, cmap='hot', aspect='auto')
    ax1.set_title(f'{title} - Full Matrix')
    ax1.set_xlabel('Residue Position')
    ax1.set_ylabel('Residue Position')
    plt.colorbar(im1, ax=ax1, label='Coupling Strength')
    
    # 2. Contact map with top contacts
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Get top contacts
    seq_len = coupling.shape[0]
    contacts = []
    for i in range(seq_len):
        for j in range(i + min_separation, seq_len):
            contacts.append((i, j, coupling[i, j]))
    
    contacts.sort(key=lambda x: x[2], reverse=True)
    top_contacts = contacts[:top_k]
    
    # Plot
    ax2.set_xlim(-0.5, seq_len - 0.5)
    ax2.set_ylim(seq_len - 0.5, -0.5)
    
    weights = [c[2] for c in top_contacts]
    max_weight = max(weights) if weights else 1.0
    
    for i, j, w in top_contacts:
        size = (w / max_weight) * 100
        ax2.scatter(j, i, s=size, c='red', alpha=0.6, edgecolors='darkred')
    
    ax2.set_title(f'Top {top_k} Contacts (sep >= {min_separation})')
    ax2.set_xlabel('Residue Position')
    ax2.set_ylabel('Residue Position')
    ax2.grid(alpha=0.3)
    
    # 3. Coupling distribution
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Get upper triangle values (excluding diagonal)
    upper_tri_indices = np.triu_indices_from(coupling, k=1)
    coupling_values = coupling[upper_tri_indices]
    
    ax3.hist(coupling_values, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Coupling Strength')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Coupling Distribution')
    ax3.axvline(np.median(coupling_values), color='red', linestyle='--', 
                label=f'Median: {np.median(coupling_values):.4f}')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, top_contacts


def compare_methods(
    logits: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Compare different coupling computation methods.
    
    Args:
        logits: ESMC logits (seq_len, vocab_size)
        save_path: Path to save comparison figure
    """
    analyzer = ESMCLogitCoupling()
    
    # Compute using different methods
    print("Computing coupling with different methods...")
    coupling_frob = analyzer.compute_coupling_from_logits(logits, method="frobenius")
    coupling_cov = analyzer.compute_coupling_from_logits(logits, method="covariance")
    coupling_mi = analyzer.compute_coupling_from_logits(logits, method="mutual_information")
    
    # Apply APC correction
    print("Applying APC correction...")
    coupling_frob_apc = analyzer.apc_correction(coupling_frob)
    coupling_cov_apc = analyzer.apc_correction(coupling_cov)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = [
        (coupling_frob, "Frobenius", axes[0, 0]),
        (coupling_cov, "Covariance", axes[0, 1]),
        (coupling_mi, "Mutual Information", axes[0, 2]),
        (coupling_frob_apc, "Frobenius + APC", axes[1, 0]),
        (coupling_cov_apc, "Covariance + APC", axes[1, 1]),
    ]
    
    for coupling, method_name, ax in methods:
        im = ax.imshow(coupling, cmap='hot', aspect='auto')
        ax.set_title(method_name)
        ax.set_xlabel('Residue Position')
        ax.set_ylabel('Residue Position')
        plt.colorbar(im, ax=ax)
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.suptitle('Comparison of Coupling Methods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage
def demo_with_mock_logits():
    """Demonstrate with mock logits."""
    print("=" * 60)
    print("ESMC Logit-Based Coupling Demo")
    print("=" * 60)
    
    # Create mock logits
    seq_len = 2234  # Your sequence length
    vocab_size = 20  # 20 amino acids
    
    print(f"\nGenerating mock logits:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    
    # Simulate logits with some structure
    np.random.seed(42)
    logits = torch.randn(seq_len, vocab_size)
    
    # Add some coupling structure
    for i in range(0, seq_len, 50):
        for j in range(i+10, min(i+60, seq_len), 20):
            # Make these positions have similar distributions
            logits[j] = logits[i] + torch.randn(vocab_size) * 0.3
    
    print("\n1. Computing coupling (Frobenius method)...")
    analyzer = ESMCLogitCoupling()
    coupling = analyzer.compute_coupling_from_logits(logits, method="frobenius")
    
    print(f"   Coupling matrix shape: {coupling.shape}")
    print(f"   Mean coupling: {coupling.mean():.4f}")
    print(f"   Max coupling: {coupling.max():.4f}")
    print(f"   Min coupling: {coupling.min():.4f}")
    
    print("\n2. Applying APC correction...")
    coupling_apc = analyzer.apc_correction(coupling)
    
    print(f"   APC coupling range: [{coupling_apc.min():.4f}, {coupling_apc.max():.4f}]")
    
    print("\n3. Computing Direct Information...")
    di = analyzer.compute_direct_information(coupling)
    
    print(f"   DI range: [{di.min():.4f}, {di.max():.4f}]")
    
    print("\n4. Creating visualizations...")
    fig1, contacts = visualize_coupling(
        coupling_apc,
        title="Evolutionary Coupling (Frobenius + APC)",
        top_k=50,
        min_separation=6,
        save_path="/home/azureuser/cloudfiles/code/Users/jc62/projects/esm3/data/h5n1_data/analysis/coupling_frobenius_apc.png"
    )
    plt.close(fig1)
    
    fig2, contacts_di = visualize_coupling(
        di,
        title="Direct Information",
        top_k=50,
        min_separation=6,
        save_path="/home/azureuser/cloudfiles/code/Users/jc62/projects/esm3/data/h5n1_data/analysis/coupling_direct_info.png"
    )
    plt.close(fig2)
    
    print("\n5. Top 10 coupled residue pairs (APC):")
    print("\n   Pos1  Pos2  Coupling")
    print("   " + "-" * 30)
    for i, j, strength in contacts[:10]:
        print(f"   {i:4d}  {j:4d}  {strength:.6f}")
    
    print("\n6. Comparing methods...")
    fig3 = compare_methods(logits, save_path="/home/azureuser/cloudfiles/code/Users/jc62/projects/esm3/data/h5n1_data/analysis/coupling_comparison.png")
    plt.close(fig3)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check outputs folder for visualizations.")
    print("=" * 60)
    
    return coupling_apc, di, contacts


if __name__ == "__main__":
    coupling, di, contacts = demo_with_mock_logits()
    
    print("\n" + "=" * 60)
    print("TO USE WITH YOUR ESMC LOGITS:")
    print("=" * 60)