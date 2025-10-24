# %%
## Import dependencies
# Standard libraries
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

# Config libraries
import yaml
from pathlib import Path
from typing import Any
from __future__ import annotations

# Biology libraries
from Bio import SeqIO
from biotite.database import rcsb

# ML libraries
import torch

# ESM3 and ESMC libraries
from esm.models.esmc import ESMC, ESMCInferenceClient, LogitsConfig, ESMProtein
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsOutput
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

# %%
## Logging in to HF and downloading the model
from huggingface_hub import login

def load_model(token: str, model_name: str):
    login(token=token)
    model: ESMCInferenceClient = ESMC.from_pretrained(model_name).to("cuda") # or "cpu"
    return model

# %%
## Loading configuration 

# Create class containing read and combine configuration
class InferenceConfig:
    def __init__(self, config_path: str | Path):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        # Loading paths
        self.input_dir = Path(self._cfg["paths"]["input_dir"])
        self.output_dir = Path(self._cfg["paths"]["output_dir"])
        self.input_path = self.input_dir / self._cfg["input"]["input_file_name"]
        self.output_path = self.output_dir / self._cfg["output"]["output_file_name"]

        # Inference parameters
        p = self._cfg["parameters"]
        self.token = p["token"]
        self.input_type = p["input_type"]
        self.model_name = p["model_name"]
        self.sequence = p["sequence"]
        self.return_embeddings = bool(p.get("return_embeddings", True))
        self.return_hidden_states = bool(p.get("return_hidden_states", True))
        self.ith_hidden_layer = p["ith_hidden_layer"]
        self.max_workers = p.get("max_workers", None)  # None = default ThreadPoolExecutor behavior

    @staticmethod
    # Get the original (resolved) YAML as a list
    def _as_list(x: Any) -> list[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [t.strip() for t in str(x).split(",") if t.strip()]

    # Or get the original (resolved) YAML as a dict
    def to_dict(self) -> dict:
        return self._cfg

# %%
## Read the FASTA file and prepare sequences
def read_fasta(fasta_path: str | Path):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id                      # e.g., 'sp|P12345|'
        seq_str = str(record.seq).upper()       # convert to string, uppercase
        seq_str = clean_sequence(seq_str)   # clean the sequence
        sequences.append((seq_id, seq_str))
        
    return sequences

## Clean the sequences (remove all non-alphabet characters)
def clean_sequence(seq: str) -> str:
    return re.sub(r"[^A-Z]", "", seq.upper())

## Convert the pdb to ESMProtein 
def convert_pdb(pdb_path: str, pdb_id: str) -> ESMProtein:
    # Check to see if pdb id provided, else fetch from RCSB
    if pdb_path is None:
        str_io = rcsb.fetch(pdb_id, "pdb")
        protein_chain = ProteinChain.from_pdb(str_io, id=pdb_id)
    else:
        protein_chain = ProteinChain.from_pdb(pdb_path)
        
    # Run the conversion
    protein = ESMProtein.from_protein_chain(
        protein_chain
        )
    return protein

## Convert the sequence to ESMProtein 
def convert_sequence(sequence: str) -> ESMProtein:
    # Clean the input sequence
    sequence = clean_sequence(sequence)
    
    # Run the conversion
    protein = ESMProtein(
        sequence=sequence,
        potential_sequence_of_concern=True
        )
    return protein

# %%
## Loading files with sequences for inference

## Load pdb lists for inference
def load_pdb(input_path: str):
    rcsb_id = pd.read_csv(input_path, header=None, names=["id"])
    pdb_id = rcsb_id["id"].tolist()
    return pdb_id

## Load fasta sequences for inference
def load_fasta(input_path: str):
    sequences = read_fasta(input_path)
    sequences_df = pd.DataFrame(sequences, columns=["id", "sequence"])
    sequences_aa_list = sequences_df["sequence"].tolist()
    return sequences_aa_list, sequences_df

# #Load csv file with sequences for inference
def load_csv(input_path: str):
    sequences_df = pd.read_csv(input_path, keep_default_na=False)
    sequences_aa_list = sequences_df['sequence'].tolist()
    return sequences_aa_list, sequences_df

# %%
## Single embedding function for one sequence
def embed_single_sequence(model, sequence: str, logits_config: LogitsConfig) -> LogitsOutput:
    """Embed a single sequence and return LogitsOutput."""
    protein = convert_sequence(sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, logits_config)
    return output

## Single embedding function for one PDB ID
def embed_single_pdb(model, pdb_id: str, logits_config: LogitsConfig) -> LogitsOutput:
    """Embed a single PDB structure and return LogitsOutput."""
    protein = convert_pdb(None, pdb_id)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, logits_config)
    return output

# %%
## Batch embedding functions
def batch_embed_sequences(
    model: ESMCInferenceClient, 
    sequences: Sequence[str],
    logits_config: LogitsConfig,
    max_workers: int = None
) -> Sequence[LogitsOutput]:
    """
    Batch embed multiple sequences using ThreadPoolExecutor for parallel processing.
    
    Args:
        model: The ESMC inference client
        sequences: List of protein sequences to embed
        logits_config: Configuration for logits output
        max_workers: Maximum number of threads (None = default)
    
    Returns:
        List of LogitsOutput objects or ESMProteinError for failed embeddings
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(embed_single_sequence, model, seq, logits_config) 
            for seq in sequences
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
    return results

def batch_embed_pdbs(
    model: ESMCInferenceClient, 
    pdb_ids: Sequence[str],
    logits_config: LogitsConfig,
    max_workers: int = None
) -> Sequence[LogitsOutput]:
    """
    Batch embed multiple PDB structures using ThreadPoolExecutor for parallel processing.
    
    Args:
        model: The ESMC inference client
        pdb_ids: List of PDB IDs to embed
        logits_config: Configuration for logits output
        max_workers: Maximum number of threads (None = default)
    
    Returns:
        List of LogitsOutput objects or ESMProteinError for failed embeddings
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(embed_single_pdb, model, pdb_id, logits_config) 
            for pdb_id in pdb_ids
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
    return results

# %%
## Load configuration and run inference at the top level
def load_and_infer(config_path: str | Path):
    # Load configuration
    cfg = InferenceConfig(config_path)

    # Load model
    model = load_model(cfg.token, cfg.model_name)

    # Create LogitsConfig object
    logits_config = LogitsConfig(
        sequence=cfg.sequence, 
        return_embeddings=cfg.return_embeddings, 
        return_hidden_states=cfg.return_hidden_states, 
        ith_hidden_layer=cfg.ith_hidden_layer
    )

    # Load data and run batch inference based on input type
    if cfg.input_type == "pdb":
        pdb_id_list = load_pdb(cfg.input_path)
        print(f"Processing {len(pdb_id_list)} PDB structures in batch mode...")
        outputs = batch_embed_pdbs(
            model=model,
            pdb_ids=pdb_id_list,
            logits_config=logits_config,
            max_workers=cfg.max_workers
        )
        ids = pdb_id_list.copy()
        
    elif cfg.input_type == "fasta":
        sequences_aa_list, sequences_df = load_fasta(cfg.input_path)
        print(f"Processing {len(sequences_aa_list)} sequences in batch mode...")
        outputs = batch_embed_sequences(
            model=model,
            sequences=sequences_aa_list,
            logits_config=logits_config,
            max_workers=cfg.max_workers
        )
        ids = sequences_df["id"].tolist()
        
    elif cfg.input_type == "csv":
        sequences_aa_list, sequences_df = load_csv(cfg.input_path)
        print(f"Processing {len(sequences_aa_list)} sequences in batch mode...")
        outputs = batch_embed_sequences(
            model=model,
            sequences=sequences_aa_list,
            logits_config=logits_config,
            max_workers=cfg.max_workers
        )
        ids = sequences_df["id"].tolist()
    else:
        print(f"Unknown input_type: {cfg.input_type}")
        return
    
    # Extract results from LogitsOutput objects, handling errors
    logits_list, embeddings_list, hidden_states_list = [], [], []
    error_count = 0
    
    for i, output in enumerate(outputs):
        if isinstance(output, ESMProteinError):
            print(f"Error processing item {i} (ID: {ids[i]}): {output}")
            error_count += 1
            # Append None or handle as needed
            logits_list.append(None)
            embeddings_list.append(None)
            hidden_states_list.append(None)
        else:
            # Extract data and move to CUDA
            logits_list.append(output.logits.sequence.squeeze(0).to("cuda", non_blocking=True))
            embeddings_list.append(output.embeddings.squeeze(0).to("cuda", non_blocking=True))
            
            hs = getattr(output, "hidden_states", None)
            if isinstance(hs, torch.Tensor):
                hidden_states_list.append(hs.squeeze().to("cuda", non_blocking=True))
            else:
                hidden_states_list.append(None)
    
    if error_count > 0:
        print(f"Warning: {error_count} out of {len(outputs)} items failed to process")
    
    # Save the outputs
    payload = {
        "id": ids,
        "logits": logits_list,
        "embeddings": embeddings_list,
        "hidden_states": hidden_states_list
    }
    torch.save(payload, cfg.output_path)
    print(f"Results saved to {cfg.output_path}")
    return 

# %%
## Top-level inference call function for calling python library
def esmc_local_embed(config_path: str | Path): 
    return load_and_infer(config_path)

# %%
# Example usage:
# esmc_local_embed("/home/azureuser/cloudfiles/code/Users/jc62/projects/esm3/code/configs/esmc_local_embed_config_v1.yaml")
