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
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)  # (seq_len, vocab_size)
        
        seq_len = probs.shape[0]
        coupling = np.zeros((seq_len, seq_len))
        
        # Compute pairwise similarity
        for i in range(seq_len):
            for j in range(i, seq_len):
                # Dot product of probability vectors
                similarity = torch.dot(probs[i], probs[j]).item()
                coupling[i, j] = similarity
                coupling[j, i] = similarity
        
        return coupling
    
    def _covariance_coupling(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute coupling based on covariance of logit vectors.
        
        High covariance suggests positions co-vary in the learned space.
        """
        # Center the logits
        logits_np = logits.cpu().numpy()
        logits_centered = logits_np - logits_np.mean(axis=1, keepdims=True)
        
        # Compute covariance matrix using outer product
        # For each pair (i,j), compute correlation between their logit profiles
        seq_len, vocab_size = logits_np.shape
        coupling = np.zeros((seq_len, seq_len))
        
        for i in range(seq_len):
            for j in range(i, seq_len):
                # Pearson correlation between logit vectors
                corr = np.corrcoef(logits_np[i], logits_np[j])[0, 1]
                coupling[i, j] = abs(corr)  # Absolute value
                coupling[j, i] = abs(corr)
        
        return coupling
    
    def _mi_coupling(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute coupling based on mutual information between predictions.
        
        Estimates MI from the predicted probability distributions.
        """
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        seq_len, vocab_size = probs.shape
        
        coupling = np.zeros((seq_len, seq_len))
        
        # For each pair, compute MI
        for i in range(seq_len):
            for j in range(i, seq_len):
                # Simple approximation: use KL divergence symmetrized
                # MI ≈ H(i) + H(j) - H(i,j)
                # We approximate this with distribution similarity
                
                # Entropy at position i
                H_i = -np.sum(probs[i] * np.log(probs[i] + 1e-10))
                # Entropy at position j  
                H_j = -np.sum(probs[j] * np.log(probs[j] + 1e-10))
                
                # Joint approximation (assume independence for normalization)
                # High coupling = deviation from independence
                joint_entropy_indep = H_i + H_j
                
                # KL divergence between distributions
                kl = np.sum(probs[i] * np.log((probs[i] + 1e-10) / (probs[j] + 1e-10)))
                
                # Mutual information proxy
                mi = H_i + H_j - abs(kl)
                
                coupling[i, j] = mi
                coupling[j, i] = mi
        
        # Normalize
        coupling = coupling / coupling.max()
        
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
    
    def compute_direct_information(
        self, 
        coupling: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute Direct Information (DI) from coupling matrix.
        
        DI removes indirect correlations using matrix inversion.
        This is similar to what DCA does.
        
        Args:
            coupling: Raw coupling matrix
            threshold: Optional threshold to zero out weak couplings
            
        Returns:
            Direct information matrix
        """
        # Add small value to diagonal for numerical stability
        coupling_reg = coupling + np.eye(coupling.shape[0]) * 1e-4
        
        # Invert to get precision matrix (removes indirect effects)
        try:
            precision = np.linalg.inv(coupling_reg)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            precision = np.linalg.pinv(coupling_reg)
        
        # DI is negative precision (off-diagonal elements)
        di = -precision
        np.fill_diagonal(di, 0)
        
        # Take absolute value and normalize
        di = np.abs(di)
        di = di / di.max()
        
        if threshold:
            di[di < threshold] = 0
        
        return di
    
    def apc_correction(self, coupling: np.ndarray) -> np.ndarray:
        """
        Apply Average Product Correction (APC).
        
        This removes phylogenetic and entropic biases.
        Standard in contact prediction.
        
        Args:
            coupling: Raw coupling matrix
            
        Returns:
            APC-corrected coupling matrix
        """
        seq_len = coupling.shape[0]
        
        # Compute row and column means
        row_means = coupling.mean(axis=1, keepdims=True)
        col_means = coupling.mean(axis=0, keepdims=True)
        total_mean = coupling.mean()
        
        # APC correction
        apc_matrix = (row_means @ col_means) / total_mean
        corrected = coupling - apc_matrix
        
        # Keep only positive values
        corrected = np.maximum(corrected, 0)
        
        return corrected


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