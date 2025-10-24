"""
ESMC Attention Analysis for Evolutionary Coupling
Extracts and visualizes attention weights between residues from ESMC model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class ESMCAttentionAnalyzer:
    """
    Analyzer for ESMC attention patterns to explore residue-residue associations.
    
    Note: This requires access to the actual ESMC model with attention outputs.
    If you only have embeddings, you'll need to re-run inference with output_attentions=True.
    """
    
    def __init__(self, model, tokenizer=None):
        """
        Initialize analyzer with ESMC model.
        
        Args:
            model: ESMC model instance
            tokenizer: ESMC tokenizer (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
    def extract_attention(
        self, 
        sequence: str, 
        layer_idx: Optional[int] = -1
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract attention weights from ESMC model.
        
        Args:
            sequence: Protein sequence string
            layer_idx: Which layer to extract (-1 for last layer, None for all)
            
        Returns:
            embeddings: (seq_len, hidden_dim) tensor
            attentions: List of (num_heads, seq_len, seq_len) tensors per layer
        """
        with torch.no_grad():
            # Tokenize and run model with attention output
            if self.tokenizer:
                inputs = self.tokenizer(sequence, return_tensors="pt")
            else:
                # Assume sequence is already tokenized
                inputs = {"input_ids": sequence}
            
            # Forward pass with attention outputs
            from esm.sdk.api import ESMProtein

            # Convert sequence string to ESMProtein
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.model.encode(protein)

            # Get attention outputs
            outputs = self.model.forward(protein_tensor, output_attentions=True)
            # outputs = self.model(**inputs, output_attentions=True)
            
            embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
            attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)
            
            # Remove batch dimension
            attentions = [attn.squeeze(0) for attn in attentions]
            
        return embeddings, attentions
    
    def aggregate_attention(
        self, 
        attentions: List[torch.Tensor], 
        method: str = "mean",
        layers: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Aggregate attention across heads and optionally layers.
        
        Args:
            attentions: List of attention tensors per layer
            method: 'mean', 'max', or 'last' (for aggregating heads)
            layers: Which layers to include (None for all)
            
        Returns:
            Aggregated attention matrix (seq_len, seq_len)
        """
        if layers is None:
            layers = list(range(len(attentions)))
        
        selected_attentions = [attentions[i] for i in layers]
        
        # Aggregate across heads first
        aggregated_layers = []
        for attn in selected_attentions:
            if method == "mean":
                agg = attn.mean(dim=0)  # Average across heads
            elif method == "max":
                agg = attn.max(dim=0)[0]  # Max across heads
            elif method == "last":
                agg = attn[-1]  # Just last head
            else:
                raise ValueError(f"Unknown method: {method}")
            aggregated_layers.append(agg)
        
        # Aggregate across layers
        final_attention = torch.stack(aggregated_layers).mean(dim=0)
        
        return final_attention.cpu().numpy()
    
    def attention_rollout(
        self, 
        attentions: List[torch.Tensor],
        start_layer: int = 0
    ) -> np.ndarray:
        """
        Compute attention rollout (cumulative attention flow).
        This better captures long-range dependencies than raw attention.
        
        Args:
            attentions: List of attention tensors per layer
            start_layer: Which layer to start rollout from
            
        Returns:
            Rolled out attention matrix (seq_len, seq_len)
        """
        # Average across heads for each layer
        attentions = [attn.mean(dim=0) for attn in attentions[start_layer:]]
        
        # Add residual connections (identity matrix)
        seq_len = attentions[0].shape[0]
        eye = torch.eye(seq_len, device=attentions[0].device)
        attentions = [0.5 * attn + 0.5 * eye for attn in attentions]
        
        # Normalize
        attentions = [attn / attn.sum(dim=-1, keepdim=True) for attn in attentions]
        
        # Rollout: multiply attention matrices
        rollout = attentions[0]
        for attn in attentions[1:]:
            rollout = torch.matmul(attn, rollout)
        
        return rollout.cpu().numpy()
    
    def compute_symmetric_attention(
        self, 
        attention: np.ndarray
    ) -> np.ndarray:
        """
        Make attention matrix symmetric by averaging A[i,j] and A[j,i].
        This is more appropriate for measuring mutual association.
        
        Args:
            attention: Asymmetric attention matrix
            
        Returns:
            Symmetric attention matrix
        """
        return (attention + attention.T) / 2
    
    def get_top_contacts(
        self, 
        attention: np.ndarray, 
        top_k: int = 20,
        min_separation: int = 6
    ) -> List[Tuple[int, int, float]]:
        """
        Get top attention contacts, filtering out local interactions.
        
        Args:
            attention: Attention matrix
            top_k: Number of top contacts to return
            min_separation: Minimum sequence separation to consider
            
        Returns:
            List of (pos_i, pos_j, attention_weight) tuples
        """
        n = attention.shape[0]
        contacts = []
        
        for i in range(n):
            for j in range(i + min_separation, n):
                contacts.append((i, j, attention[i, j]))
        
        # Sort by attention weight
        contacts.sort(key=lambda x: x[2], reverse=True)
        
        return contacts[:top_k]


def visualize_attention_matrix(
    attention: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Attention Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Visualize attention matrix as a heatmap.
    
    Args:
        attention: Attention matrix (seq_len, seq_len)
        sequence: Optional protein sequence for axis labels
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        vmin, vmax: Color scale limits
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attention, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Labels
    ax.set_xlabel('Residue Position (j)')
    ax.set_ylabel('Residue Position (i)')
    ax.set_title(title)
    
    # Add sequence labels if provided (only for small sequences)
    if sequence and len(sequence) < 100:
        ax.set_xticks(range(len(sequence)))
        ax.set_yticks(range(len(sequence)))
        ax.set_xticklabels(list(sequence), fontsize=6)
        ax.set_yticklabels(list(sequence), fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def visualize_attention_contacts(
    attention: np.ndarray,
    top_k: int = 50,
    min_separation: int = 6,
    sequence: Optional[str] = None,
    title: str = "Top Attention Contacts",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Visualize top attention contacts as a contact map.
    
    Args:
        attention: Attention matrix
        top_k: Number of top contacts to show
        min_separation: Minimum sequence separation
        sequence: Optional sequence
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    # Get symmetric attention
    sym_attention = (attention + attention.T) / 2
    
    # Get top contacts
    n = attention.shape[0]
    contacts = []
    
    for i in range(n):
        for j in range(i + min_separation, n):
            contacts.append((i, j, sym_attention[i, j]))
    
    contacts.sort(key=lambda x: x[2], reverse=True)
    top_contacts = contacts[:top_k]
    
    # Create contact map
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background
    background = np.zeros((n, n))
    ax.imshow(background, cmap='gray_r', alpha=0.3, aspect='auto')
    
    # Plot contacts with size proportional to weight
    weights = [c[2] for c in top_contacts]
    max_weight = max(weights)
    
    for i, j, w in top_contacts:
        size = (w / max_weight) * 100
        ax.scatter(j, i, s=size, c='red', alpha=0.6, edgecolors='darkred')
        ax.scatter(i, j, s=size, c='red', alpha=0.6, edgecolors='darkred')  # Symmetric
    
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Residue Position')
    ax.set_title(f'{title} (Top {top_k}, min_sep={min_separation})')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # Invert y-axis
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def visualize_attention_profile(
    attention: np.ndarray,
    positions: List[int],
    sequence: Optional[str] = None,
    title: str = "Attention Profile",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Visualize attention profile from specific positions.
    
    Args:
        attention: Attention matrix
        positions: List of positions to show profiles for
        sequence: Optional sequence
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(len(positions), 1, figsize=figsize, 
                             sharex=True, squeeze=False)
    
    seq_len = attention.shape[0]
    x = np.arange(seq_len)
    
    for idx, pos in enumerate(positions):
        ax = axes[idx, 0]
        
        # Plot attention from this position
        ax.plot(x, attention[pos, :], linewidth=2, color='steelblue')
        ax.axvline(pos, color='red', linestyle='--', alpha=0.5, label=f'Query pos {pos}')
        
        ax.set_ylabel('Attention')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        if sequence and pos < len(sequence):
            ax.set_title(f'Position {pos} ({sequence[pos]})')
    
    axes[-1, 0].set_xlabel('Residue Position')
    fig.suptitle(title, y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


# Example usage demonstration
def demo_with_mock_data():
    """
    Demonstrate the analysis with mock data.
    Replace this with actual ESMC model inference.
    """
    print("=" * 60)
    print("ESMC Attention Analysis Demo")
    print("=" * 60)
    
    # Create mock attention data (in practice, extract from ESMC)
    seq_len = 100
    num_layers = 12
    num_heads = 20
    
    print(f"\nGenerating mock attention for:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Attention heads per layer: {num_heads}")
    
    # Simulate attention with some structure
    np.random.seed(42)
    attentions = []
    
    for layer in range(num_layers):
        # Create attention with diagonal + some random long-range contacts
        layer_attn = []
        for head in range(num_heads):
            attn = np.random.exponential(0.1, (seq_len, seq_len))
            # Add stronger local interactions
            for i in range(seq_len):
                for j in range(max(0, i-5), min(seq_len, i+6)):
                    attn[i, j] += np.random.exponential(0.5)
            # Normalize
            attn = attn / attn.sum(axis=1, keepdims=True)
            layer_attn.append(attn)
        
        attentions.append(np.array(layer_attn))
    
    # Aggregate attention (mean across heads and last 4 layers)
    print("\n1. Aggregating attention (mean across heads, last 4 layers)...")
    selected_layers = attentions[-4:]
    aggregated = np.stack([layer.mean(axis=0) for layer in selected_layers]).mean(axis=0)
    
    # Make symmetric
    sym_attention = (aggregated + aggregated.T) / 2
    
    print(f"   Attention matrix shape: {sym_attention.shape}")
    print(f"   Mean attention weight: {sym_attention.mean():.4f}")
    print(f"   Max attention weight: {sym_attention.max():.4f}")
    
    # Visualizations
    print("\n2. Creating visualizations...")
    
    # Full attention matrix
    print("   - Attention heatmap...")
    fig1, _ = visualize_attention_matrix(
        sym_attention,
        title="Symmetric Attention Matrix (Last 4 Layers, All Heads)",
        save_path="/mnt/user-data/outputs/attention_heatmap.png"
    )
    plt.close(fig1)
    
    # Top contacts
    print("   - Top contacts map...")
    fig2, _ = visualize_attention_contacts(
        sym_attention,
        top_k=50,
        min_separation=6,
        title="Top 50 Attention Contacts",
        save_path="/mnt/user-data/outputs/attention_contacts.png"
    )
    plt.close(fig2)
    
    # Attention profiles
    print("   - Attention profiles...")
    fig3, _ = visualize_attention_profile(
        sym_attention,
        positions=[10, 25, 50, 75],
        title="Attention Profiles from Selected Positions",
        save_path="/mnt/user-data/outputs/attention_profiles.png"
    )
    plt.close(fig3)
    
    # Identify top contacts
    print("\n3. Top attention contacts (sequence separation >= 6):")
    contacts = []
    for i in range(seq_len):
        for j in range(i + 6, seq_len):
            contacts.append((i, j, sym_attention[i, j]))
    
    contacts.sort(key=lambda x: x[2], reverse=True)
    
    print("\n   Pos1  Pos2  Attention")
    print("   " + "-" * 25)
    for i, j, weight in contacts[:10]:
        print(f"   {i:4d}  {j:4d}  {weight:.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check outputs folder for visualizations.")
    print("=" * 60)
    
    return sym_attention, contacts


if __name__ == "__main__":
    # Run demo with mock data
    attention, contacts = demo_with_mock_data()
    
    print("\n" + "=" * 60)
    print("TO USE WITH REAL ESMC MODEL:")
    print("=" * 60)
    print("""
# 1. Load ESMC model
from esm.models import ESMC  # or your import path
model = ESMC.from_pretrained("esmc_600m")

# 2. Initialize analyzer
analyzer = ESMCAttentionAnalyzer(model, tokenizer)

# 3. Extract attention
sequence = "MKTAYIAKQR..."  # your protein sequence
embeddings, attentions = analyzer.extract_attention(sequence)

# 4. Aggregate attention
attention_matrix = analyzer.aggregate_attention(
    attentions, 
    method="mean",
    layers=[-4, -3, -2, -1]  # Last 4 layers
)

# 5. Make symmetric
sym_attention = analyzer.compute_symmetric_attention(attention_matrix)

# 6. Visualize
visualize_attention_matrix(sym_attention, sequence)
visualize_attention_contacts(sym_attention, top_k=50)

# 7. Get top contacts
contacts = analyzer.get_top_contacts(sym_attention, top_k=20)
    """)