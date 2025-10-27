# %%
import pandas as pd
import numpy as np
import re
import sys
import torch

# %%
def score_mutation_importance(
    position, 
    entropy, 
    coupling_matrix,
    other_mutations
):
    """
    Score importance of a mutation based on context.
    
    Returns:
        importance_score: Higher = more critical/context-dependent
        independence_score: Higher = more independent/robust
    """
    
    # 1. Constraint score (inverse of entropy)
    constraint = 1.0 / (entropy[position] + 0.1)
    
    # 2. Coupling to other adaptive mutations
    coupling_to_adaptive = coupling_matrix[position, other_mutations].mean()
    
    # 3. Overall coupling strength
    overall_coupling = coupling_matrix[position, :].mean()
    
    # Importance score
    importance = constraint * coupling_to_adaptive
    
    # Independence score
    independence = constraint / (overall_coupling + 0.1)
    
    return {
        'constraint': constraint,
        'coupling_to_adaptive': coupling_to_adaptive,
        'overall_coupling': overall_coupling,
        'importance': importance,
        'independence': independence
    }


def score_mutation_importance_gpu(
    position, 
    entropy, 
    coupling_matrix,
    other_mutations,
    device='cuda'
):
    """
    GPU-accelerated version: Score importance of a mutation based on context.
    
    Args:
        position: Integer position of mutation
        entropy: Torch tensor of shape (num_residues,)
        coupling_matrix: Torch tensor of shape (num_residues, num_residues)
        other_mutations: List or tensor of mutation positions
        device: 'cuda' or 'cpu'
    
    Returns:
        importance_score: Higher = more critical/context-dependent
        independence_score: Higher = more independent/robust
    """
    # Ensure tensors are on GPU
    entropy = entropy.to(device)
    coupling_matrix = coupling_matrix.to(device)
    
    # Convert other_mutations to tensor if needed
    if isinstance(other_mutations, list):
        other_mutations = torch.tensor(other_mutations, device=device)
    else:
        other_mutations = other_mutations.to(device)
    
    # 1. Constraint score (inverse of entropy)
    constraint = 1.0 / (entropy[position] + 0.1)
    
    # 2. Coupling to other adaptive mutations (GPU-accelerated)
    coupling_to_adaptive = coupling_matrix[position, other_mutations].mean()
    
    # 3. Overall coupling strength (GPU-accelerated)
    overall_coupling = coupling_matrix[position, :].mean()
    
    # Importance score
    importance = constraint * coupling_to_adaptive
    
    # Independence score
    independence = constraint / (overall_coupling + 0.1)
    
    return {
        'constraint': constraint.item(),
        'coupling_to_adaptive': coupling_to_adaptive.item(),
        'overall_coupling': overall_coupling.item(),
        'importance': importance.item(),
        'independence': independence.item()
    }


def score_mutations_batch_gpu(
    positions,
    entropy,
    coupling_matrix,
    other_mutations_list,
    device='cuda'
):
    """
    GPU-accelerated batch scoring of multiple mutations.
    
    Args:
        positions: Tensor or list of mutation positions
        entropy: Torch tensor of shape (num_residues,)
        coupling_matrix: Torch tensor of shape (num_residues, num_residues)
        other_mutations_list: List of lists, each containing mutation positions
        device: 'cuda' or 'cpu'
    
    Returns:
        results: List of dicts with scores for each mutation
    """
    # Ensure tensors are on GPU
    entropy = entropy.to(device)
    coupling_matrix = coupling_matrix.to(device)
    
    if isinstance(positions, list):
        positions = torch.tensor(positions, device=device)
    else:
        positions = positions.to(device)
    
    results = []
    
    for i, position in enumerate(positions):
        pos = position.item() if isinstance(position, torch.Tensor) else position
        other_mutations = other_mutations_list[i] if isinstance(other_mutations_list[i], torch.Tensor) else torch.tensor(other_mutations_list[i], device=device)
        
        score = score_mutation_importance_gpu(pos, entropy, coupling_matrix, other_mutations, device)
        score['position'] = pos
        results.append(score)
    
    return results


def score_all_mutations_gpu(
    entropy,
    coupling_matrix,
    threshold_entropy=None,
    threshold_coupling=None,
    batch_size=1000,
    device='cuda'
):
    """
    GPU-accelerated scoring of all mutations (vectorized).
    
    Args:
        entropy: Torch tensor of shape (num_residues,)
        coupling_matrix: Torch tensor of shape (num_residues, num_residues)
        threshold_entropy: Only score positions with entropy below this threshold (optional)
        threshold_coupling: Only score positions with coupling above this threshold (optional)
        batch_size: Number of mutations to score per batch
        device: 'cuda' or 'cpu'
    
    Returns:
        scores: Dict with tensors of shape (num_scored_positions,):
            - constraint: Constraint scores
            - coupling_to_adaptive: Coupling scores
            - overall_coupling: Overall coupling
            - importance: Importance scores
            - independence: Independence scores
            - positions: Positions that were scored
    """
    # Move to GPU
    entropy = entropy.to(device).float()
    coupling_matrix = coupling_matrix.to(device).float()
    
    num_residues = entropy.shape[0]
    
    # Filter positions if thresholds provided
    if threshold_entropy is not None:
        mask = entropy < threshold_entropy
        positions = torch.where(mask)[0]
    elif threshold_coupling is not None:
        overall_coupling = coupling_matrix.mean(dim=1)
        mask = overall_coupling > threshold_coupling
        positions = torch.where(mask)[0]
    else:
        positions = torch.arange(num_residues, device=device)
    
    # Vectorized computation for all positions
    constraint = 1.0 / (entropy + 0.1)
    overall_coupling = coupling_matrix.mean(dim=1)
    
    # For coupling_to_adaptive, use row mean (approximate - all positions as "other mutations")
    coupling_to_adaptive = overall_coupling  # or use specific subset if needed
    
    # Calculate scores
    importance = constraint * coupling_to_adaptive
    independence = constraint / (overall_coupling + 0.1)
    
    return {
        'constraint': constraint,
        'coupling_to_adaptive': coupling_to_adaptive,
        'overall_coupling': overall_coupling,
        'importance': importance,
        'independence': independence,
        'positions': positions,
        'num_scored': positions.shape[0]
    }


def convert_to_gpu_tensors(entropy, coupling_matrix, device='cuda'):
    """
    Convert numpy arrays to GPU tensors.
    
    Args:
        entropy: Numpy array or torch tensor
        coupling_matrix: Numpy array or torch tensor
        device: 'cuda' or 'cpu'
    
    Returns:
        entropy_gpu: Torch tensor on GPU
        coupling_gpu: Torch tensor on GPU
    """
    if isinstance(entropy, np.ndarray):
        entropy = torch.from_numpy(entropy).float()
    else:
        entropy = entropy.float()
    
    if isinstance(coupling_matrix, np.ndarray):
        coupling_matrix = torch.from_numpy(coupling_matrix).float()
    else:
        coupling_matrix = coupling_matrix.float()
    
    entropy = entropy.to(device)
    coupling_matrix = coupling_matrix.to(device)
    
    return entropy, coupling_matrix


def clear_mutation_cache():
    """Clear GPU cache to free up memory."""
    torch.cuda.empty_cache()

# %%
''''
**Mutation Classification**

Based on entropy and coupling, classify mutations:
```
┌────────────────────┬──────────────────┬──────────────────────┐
│                    │  Low Entropy     │  High Entropy        │
│                    │  (Constrained)   │  (Flexible)          │
├────────────────────┼──────────────────┼──────────────────────┤
│ Low Coupling       │ Type 1:          │ Type 2:              │
│ (Independent)      │ Critical Core    │ Surface Tuning       │
│                    │ → Active site    │ → Fine-tuning        │
│                    │ → Catalytic      │ → Less important     │
├────────────────────┼──────────────────┼──────────────────────┤
│ High Coupling      │ Type 3:          │ Type 4:              │
│ (Context-dep)      │ Structural Hub   │ Flexible Interface   │
│                    │ → Critical       │ → Conditional        │
│                    │ → Needs network  │ → Context-specific   │
└────────────────────┴──────────────────┴──────────────────────┘
'''