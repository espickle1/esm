# %%
## Libraries
import torch
import numpy as np

# %%
## Calculate the entropy
def calculate_shannon_entropy(logits, base='e'):
    """
    Calculate Shannon entropy for each residue from logits.
    
    Args:
        logits: Tensor of shape (num_residues, num_tokens)
                e.g., (2234, 20) for amino acids
        base: 'e' (nats), '2' (bits), or '10' (dits)
    
    Returns:
        entropy: Tensor of shape (num_residues,) 
        
    Formula: H(X) = -Σ p(x) * log(p(x))
    """
    # Convert logits to probabilities (stays on GPU)
    probs = torch.softmax(logits, dim=-1)
    
    # Calculate entropy using torch operations
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)
    entropy = -torch.sum(probs * log_probs, dim=1)
    
    # Convert base if needed
    if base == '2':
        entropy = entropy / np.log(2)  # bits
    elif base == '10':
        entropy = entropy / np.log(10)  # dits

    return entropy


def calculate_shannon_entropy_batched(logits, base='e', batch_size=10000):
    """
    Calculate Shannon entropy for very large tensors using batching.
    
    Args:
        logits: Tensor of shape (num_residues, num_tokens)
        base: 'e' (nats), '2' (bits), or '10' (dits)
        batch_size: Number of residues to process per batch (adjust based on GPU memory)
    
    Returns:
        entropy: Tensor of shape (num_residues,)
    """
    num_residues = logits.shape[0]
    device = logits.device
    entropy_list = []
    
    for i in range(0, num_residues, batch_size):
        batch_end = min(i + batch_size, num_residues)
        logits_batch = logits[i:batch_end]
        
        # Process batch
        probs_batch = torch.softmax(logits_batch, dim=-1)
        epsilon = 1e-10
        log_probs_batch = torch.log(probs_batch + epsilon)
        entropy_batch = -torch.sum(probs_batch * log_probs_batch, dim=1)
        
        entropy_list.append(entropy_batch)
    
    # Concatenate all batches
    entropy = torch.cat(entropy_list, dim=0)
    
    # Convert base if needed
    if base == '2':
        entropy = entropy / np.log(2)
    elif base == '10':
        entropy = entropy / np.log(10)
    
    return entropy


def calculate_shannon_entropy_mixed_precision(logits, base='e', batch_size=10000):
    """
    Calculate Shannon entropy using mixed precision (fp16) for faster computation on large tensors.
    
    Args:
        logits: Tensor of shape (num_residues, num_tokens)
        base: 'e' (nats), '2' (bits), or '10' (dits)
        batch_size: Number of residues to process per batch
    
    Returns:
        entropy: Tensor of shape (num_residues,) in fp32
    """
    num_residues = logits.shape[0]
    device = logits.device
    entropy_list = []
    
    # Cast to fp16 for faster computation
    logits_fp16 = logits.half()
    
    for i in range(0, num_residues, batch_size):
        batch_end = min(i + batch_size, num_residues)
        logits_batch = logits_fp16[i:batch_end]
        
        # Process batch in fp16
        probs_batch = torch.softmax(logits_batch, dim=-1)
        epsilon = 1e-10
        log_probs_batch = torch.log(probs_batch + epsilon)
        entropy_batch = -torch.sum(probs_batch * log_probs_batch, dim=1)
        
        # Convert back to fp32 for accuracy
        entropy_list.append(entropy_batch.float())
    
    # Concatenate all batches
    entropy = torch.cat(entropy_list, dim=0)
    
    # Convert base if needed
    if base == '2':
        entropy = entropy / np.log(2)
    elif base == '10':
        entropy = entropy / np.log(10)
    
    return entropy

# %%
# Running the entropy calculation
def run_entropy_calculation(logits, base):
    entropy = calculate_shannon_entropy(logits, base='e')
    
    # Ensure entropy is float dtype for quantile operations
    entropy = entropy.float()
    
    # Find constrained positions (low entropy) using torch operations
    p10 = torch.quantile(entropy, 0.1)
    constrained = torch.where(entropy < p10)[0]
    
    # Find flexible positions (high entropy) using torch operations
    p90 = torch.quantile(entropy, 0.9)
    flexible = torch.where(entropy > p90)[0]

    return entropy, constrained, flexible


def run_entropy_calculation_batched(logits, base, batch_size=10000, use_mixed_precision=False):
    """
    Run entropy calculation with batching for very large tensors.
    
    Args:
        logits: Tensor of shape (num_residues, num_tokens)
        base: 'e' (nats), '2' (bits), or '10' (dits)
        batch_size: Number of residues per batch (default 10000, adjust based on GPU memory)
        use_mixed_precision: If True, use fp16 for faster computation
    
    Returns:
        entropy: Tensor of shape (num_residues,)
        constrained: Indices of low entropy positions (bottom 10%)
        flexible: Indices of high entropy positions (top 10%)
    """
    # Choose calculation method
    if use_mixed_precision:
        entropy = calculate_shannon_entropy_mixed_precision(logits, base='e', batch_size=batch_size)
    else:
        entropy = calculate_shannon_entropy_batched(logits, base='e', batch_size=batch_size)
    
    # Ensure entropy is float dtype for quantile operations
    entropy = entropy.float()
    
    # Find constrained positions (low entropy) using torch operations
    p10 = torch.quantile(entropy, 0.1)
    constrained = torch.where(entropy < p10)[0]
    
    # Find flexible positions (high entropy) using torch operations
    p90 = torch.quantile(entropy, 0.9)
    flexible = torch.where(entropy > p90)[0]

    return entropy, constrained, flexible


def calculate_entropy_for_proteins(protein_logits_list, base='e', protein_batch_size=50, 
                                   use_mixed_precision=False, return_per_protein=False):
    """
    Calculate Shannon entropy for multiple proteins efficiently.
    
    Args:
        protein_logits_list: List of tensors, each of shape (num_residues_i, num_tokens)
                             e.g., list of 1370 proteins, each (2500, 20)
        base: 'e' (nats), '2' (bits), or '10' (dits)
        protein_batch_size: Number of proteins to stack and process together (default 50)
        use_mixed_precision: If True, use fp16 for faster computation
        return_per_protein: If True, return list of per-protein entropies and statistics
    
    Returns:
        If return_per_protein=False:
            all_entropy: Concatenated entropy for all residues
            global_constrained: Global low entropy residue indices
            global_flexible: Global high entropy residue indices
        
        If return_per_protein=True:
            protein_results: List of dicts with per-protein entropy and statistics
    """
    num_proteins = len(protein_logits_list)
    all_entropy = []
    protein_results = []
    
    device = protein_logits_list[0].device
    
    # Process proteins in batches
    for batch_start in range(0, num_proteins, protein_batch_size):
        batch_end = min(batch_start + protein_batch_size, num_proteins)
        batch_proteins = protein_logits_list[batch_start:batch_end]
        
        # Stack proteins into single tensor for efficient GPU processing
        stacked_logits = torch.cat(batch_proteins, dim=0)
        
        # Calculate entropy for this batch of proteins
        if use_mixed_precision:
            batch_entropy = calculate_shannon_entropy_mixed_precision(
                stacked_logits, base=base, batch_size=10000
            )
        else:
            batch_entropy = calculate_shannon_entropy_batched(
                stacked_logits, base=base, batch_size=10000
            )
        
        # Store results and track per-protein statistics if needed
        start_idx = 0
        for i, protein_logits in enumerate(batch_proteins):
            num_residues = protein_logits.shape[0]
            end_idx = start_idx + num_residues
            
            protein_entropy = batch_entropy[start_idx:end_idx]
            all_entropy.append(protein_entropy)
            
            if return_per_protein:
                protein_results.append({
                    'protein_idx': batch_start + i,
                    'entropy': protein_entropy,
                    'mean_entropy': protein_entropy.mean().item(),
                    'std_entropy': protein_entropy.std().item(),
                    'min_entropy': protein_entropy.min().item(),
                    'max_entropy': protein_entropy.max().item(),
                })
            
            start_idx = end_idx
        
        # Clear cache between batches
        torch.cuda.empty_cache()
    
    # Concatenate all entropies
    all_entropy = torch.cat(all_entropy, dim=0)
    
    if return_per_protein:
        return protein_results, all_entropy
    else:
        # Calculate global statistics
        all_entropy_float = all_entropy.float()
        p10 = torch.quantile(all_entropy_float, 0.1)
        p90 = torch.quantile(all_entropy_float, 0.9)
        
        global_constrained = torch.where(all_entropy_float < p10)[0]
        global_flexible = torch.where(all_entropy_float > p90)[0]
        
        return all_entropy, global_constrained, global_flexible


def estimate_batch_size(num_residues, num_tokens, gpu_memory_gb=8):
    """
    Estimate appropriate batch size based on tensor dimensions and available GPU memory.
    
    Args:
        num_residues: Number of residues (rows)
        num_tokens: Number of token types (columns)
        gpu_memory_gb: Available GPU memory in GB (default 8GB)
    
    Returns:
        batch_size: Recommended batch size
    """
    # Each element takes 4 bytes (fp32) or 2 bytes (fp16)
    # We need space for: logits, probs, log_probs intermediate tensors
    # Rough estimate: 4 tensors worth of space needed
    bytes_per_element_fp32 = 4
    intermediate_factor = 4  # logits, probs, log_probs, entropy
    
    available_bytes = gpu_memory_gb * 1e9 * 0.8  # Use 80% of available memory
    bytes_per_residue = num_tokens * bytes_per_element_fp32 * intermediate_factor
    
    batch_size = int(available_bytes / bytes_per_residue)
    
    # Cap at reasonable values
    batch_size = max(1000, min(batch_size, 50000))
    
    return batch_size


def estimate_protein_batch_size(num_proteins_per_batch, residues_per_protein, num_tokens, gpu_memory_gb=8):
    """
    Estimate optimal number of proteins to process together.
    
    Args:
        num_proteins_per_batch: Number of proteins to stack (e.g., 50, 100)
        residues_per_protein: Average residues per protein (e.g., 2500)
        num_tokens: Number of token types (e.g., 20 for amino acids)
        gpu_memory_gb: Available GPU memory in GB (default 8GB)
    
    Returns:
        protein_batch_size: Recommended number of proteins to process together
    """
    bytes_per_element_fp32 = 4
    intermediate_factor = 4  # storage for intermediate tensors
    
    available_bytes = gpu_memory_gb * 1e9 * 0.8
    bytes_per_protein = residues_per_protein * num_tokens * bytes_per_element_fp32 * intermediate_factor
    
    protein_batch_size = int(available_bytes / bytes_per_protein)
    
    # Cap at reasonable values
    protein_batch_size = max(10, min(protein_batch_size, 200))
    
    return protein_batch_size


def clear_gpu_cache():
    """Clear GPU cache to free up memory between operations."""
    torch.cuda.empty_cache()


def extract_protein_logits_from_dict(protein_dict_list, device='cuda'):
    """
    Extract logits from a list of protein dictionaries and prepare for batch processing.
    
    Args:
        protein_dict_list: List of dictionaries, each containing 'logits' key
                          e.g., [{'logits': tensor1}, {'logits': tensor2}, ...]
        device: Device to move tensors to ('cuda' or 'cpu')
    
    Returns:
        protein_logits_list: List of logit tensors on specified device
        protein_info: Dict with info about proteins (lengths, valid indices)
    """
    protein_logits_list = []
    protein_info = {
        'lengths': [],
        'valid_indices': [],
        'num_proteins': len(protein_dict_list),
    }
    
    for idx, protein_dict in enumerate(protein_dict_list):
        if 'logits' in protein_dict:
            logits = protein_dict['logits']
            
            # Convert to tensor if it's numpy array
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits).float()
            
            # Ensure tensor and move to device
            logits = logits.to(device).float()
            
            # Store logits and info
            protein_logits_list.append(logits)
            protein_info['lengths'].append(logits.shape[0])
            protein_info['valid_indices'].append(idx)
        else:
            print(f"Warning: Protein {idx} missing 'logits' key, skipping...")
    
    protein_info['num_valid_proteins'] = len(protein_logits_list)
    protein_info['total_residues'] = sum(protein_info['lengths'])
    protein_info['avg_length'] = protein_info['total_residues'] / protein_info['num_valid_proteins']
    
    return protein_logits_list, protein_info


def extract_from_polymerase_dict(polymerase_list, device='cuda'):
    """
    Extract logits from polymerase dictionary structure.
    
    Args:
        polymerase_list: List of dicts, each with structure like {'logits': tensor}
                        e.g., list of 1370 polymerase records
        device: Device to move tensors to ('cuda' or 'cpu')
    
    Returns:
        protein_logits_list: List of logit tensors
        protein_info: Dict with statistics
    
    Example:
        polymerase_data = [polymerase_1, polymerase_2, ...]
        logits_list, info = extract_from_polymerase_dict(polymerase_data, device='cuda')
        print(f"Loaded {info['num_valid_proteins']} proteins")
        print(f"Total residues: {info['total_residues']}")
        print(f"Avg protein length: {info['avg_length']:.0f}")
    """
    return extract_protein_logits_from_dict(polymerase_list, device=device)
    