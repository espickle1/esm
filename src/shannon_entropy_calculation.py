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
        entropy: Array of shape (num_residues,) 
        
    Formula: H(X) = -Σ p(x) * log(p(x))
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    probs_np = probs.cpu().numpy()
    
    # Calculate entropy
    epsilon = 1e-10
    log_probs = np.log(probs_np + epsilon)
    entropy = -np.sum(probs_np * log_probs, axis=1)
    
    # Convert base if needed
    if base == '2':
        entropy = entropy / np.log(2)  # bits
    elif base == '10':
        entropy = entropy / np.log(10)  # dits

    return entropy

# %%
# Running the entropy calculation
def run_entropy_calculation(logits, base):
    entropy = calculate_shannon_entropy(logits, base='e')
    # Find constrained positions (low entropy)
    constrained = np.where(entropy < np.percentile(entropy, 10))[0]
    # 
    # Find flexible positions (high entropy)
    flexible = np.where(entropy > np.percentile(entropy, 90))[0]

    return entropy, constrained, flexible
    