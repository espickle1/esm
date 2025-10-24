# %%
## Import dependencies
# Standard libraries
import pandas as pd
import re

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
from esm.sdk.api import ESMProtein
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
## Load configuration and run inference at the top level
def load_and_infer(config_path: str | Path):
    # Load configuration
    cfg = InferenceConfig(config_path)

    # Load model
    model = load_model(cfg.token, cfg.model_name)

    ## ESMC inference function for list of pdb IDs or sequences
    def inference_list_combined(model, pdb_id_list, sequences_list):
        logits_list, embeddings_list, hidden_states_list = [], [], []

        ## Input sequences and output logits and embeddings
        def logits_output_generator(model, protein: ESMProtein):
            # Encode the input ESMProtein to tensor
            protein_tensor = model.encode(protein)

            # Run inference and output logits, embeddings, and hidden states
            logits_output = model.logits(
            protein_tensor, LogitsConfig(
                sequence=cfg.sequence, 
                return_embeddings=cfg.return_embeddings, 
                return_hidden_states=cfg.return_hidden_states, 
                ith_hidden_layer=cfg.ith_hidden_layer
                )
            )
            return logits_output

        # Inference for sequences input
        if pdb_id_list is None and sequences_list is not None:
            for seq in sequences_list:
                out = logits_output_generator(model, convert_sequence(seq))

                logits_list.append(out.logits.sequence.squeeze(0).to("cuda", non_blocking=True))
                embeddings_list.append(out.embeddings.squeeze(0).to("cuda", non_blocking=True))

                hs = getattr(out, "hidden_states", None)
                if isinstance(hs, torch.Tensor):
                    hidden_states_list.append(hs.squeeze().to("cuda", non_blocking=True))
        
        # Inference for pdb ID input
        elif pdb_id_list is not None and sequences_list is None:
            for pdb_id in pdb_id_list:
                out = logits_output_generator(model, convert_pdb(None, pdb_id))

                logits_list.append(out.logits.sequence.squeeze(0).to("cuda", non_blocking=True))
                embeddings_list.append(out.embeddings.squeeze(0).to("cuda", non_blocking=True))

                hs = getattr(out, "hidden_states", None)
                if isinstance(hs, torch.Tensor):
                    hidden_states_list.append(hs.squeeze().to("cuda", non_blocking=True))
        return logits_list, embeddings_list, hidden_states_list

    # Load data and run inference based on input type
    if cfg.input_type == "pdb":
        pdb_id_list = load_pdb(cfg.input_path)
        logits, embeddings, hidden_states = inference_list_combined(
            model=model,
            pdb_id_list=pdb_id_list, 
            sequences_list=None
            )
        ids = pdb_id_list.copy()
    elif cfg.input_type == "fasta":
        sequences_aa_list, sequences_df = load_fasta(cfg.input_path)
        logits, embeddings, hidden_states = inference_list_combined(
            model=model,
            pdb_id_list=None, 
            sequences_list=sequences_aa_list
            )
        ids = sequences_df["id"].tolist()
    elif cfg.input_type == "csv":
        sequences_aa_list, sequences_df = load_csv(cfg.input_path)
        logits, embeddings, hidden_states = inference_list_combined(
            model=model,
            pdb_id_list=None, 
            sequences_list=sequences_aa_list
            )
        ids = sequences_df["id"].tolist()
    else:
        return
    
    # Save the outputs
    payload = {
        "id": ids,
        "logits": logits,
        "embeddings": embeddings,
        "hidden_states": hidden_states
    }
    torch.save(payload, cfg.output_path)
    return 

# %%
## Top-level inference call function for calling python library
def esmc_local_embed(config_path: str | Path): 
    return load_and_infer(config_path)

# %%
esmc_local_embed("/home/azureuser/cloudfiles/code/Users/jc62/projects/esm3/code/configs/esmc_local_embed_config_v1.yaml")


