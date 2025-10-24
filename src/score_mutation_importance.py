# %%
import pandas as pd
import numpy as np
import re
import sys

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