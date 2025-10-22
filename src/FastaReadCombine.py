# %%
## Import libraries
import pandas as pd
import re
from Bio.SeqIO.FastaIO import SimpleFastaParser
import yaml
from pathlib import Path
from typing import Any

# %%
## Loading configuration 

# Create class containing read and combine configuration
class ReadCombineConfig:
    def __init__(self, config_path: str | Path):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        # Loading paths
        self.base_dir = Path(self._cfg["paths"]["base_dir"])
        self.fasta_path = self.base_dir / self._cfg["input"]["fasta_file_name"]
        self.output_path = self.base_dir / self._cfg["output"]["output_file_name"]

        # Loading parameters
        p = self._cfg["parameters"]
        self.column_names = self._as_list(p["column_names"])
        self.allowed = self._as_list(p["allowed"])
        self.order = p["order"]
        self.output_name = p["output_name"]
        self.sequence_sort_order = self._as_list(p["sequence_sort_order"])
        self.columns_for_new_id = self._as_list(p["columns_for_new_id"])

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
## Read sequence data

# Read fasta file into a dataframe
def read_fasta_to_dataframe(fasta_path):
    records = []
    append = records.append
    with open(fasta_path) as fh:
        for title, seq in SimpleFastaParser(fh):
            
            # title is the full header (without '>'); id = first token
            rec_id = title.split(None, 1)[0]

            # Clean sequence as each sequence is read
            append({"id": rec_id, "sequence": clean_sequence(seq)})
    return pd.DataFrame.from_records(records)

# Clean up fasta as needed
_clean = re.compile(r"[^A-Za-z]") 
def clean_sequence(s: str) -> str:
    return _clean.sub("", s)

# %%
## Processing fasta

# Split the fasta information in the id column
def split_by_id(sequence, columns):
    parts = sequence['id'].str.split('|', expand=True)

    # Keep only as many columns as we were asked to create, and name them
    parts = parts.iloc[:, :len(columns)]
    parts.columns = columns

    # Force string dtype (no numeric coercion) and replace true-missing with "NA"
    parts = parts.astype("string[python]").fillna("NA")
    return pd.concat([sequence, parts], axis=1)

# Merge protein sequences by categories as criteria
def merge_protein_sequences(
        sequence, 
        output_name, 
        allowed, order, 
        sequence_sort_order, 
        new_id_name
        ):
    subset = sequence.sort_values(by=sequence_sort_order)
    
    # Keep only allowed; if duplicates per (isolate, protein), keep first
    tmp = (subset[subset["protein"].isin(allowed)]
           .sort_values(["isolate", "protein"], key=lambda s: s.map(order))
           .drop_duplicates(["isolate", "protein"], keep="first"))

    # For each isolate, concatenate the target column in order given
    agg = (tmp.groupby("isolate", as_index=False)
              .agg(sequence=("sequence", lambda s: "".join(s.astype(str)))))

    # Build the new rows with the name column
    new_rows = agg.assign(protein=output_name)
    
    # Append to the original dataframe
    out = pd.concat([subset, new_rows], ignore_index=True)
    out = fill_metadata(out, new_id_name, output_name)
    return out

# Fill out NA entries in the metadata after merging 
def fill_metadata(out, new_id_name, output_name):
    cols_to_fill = [c for c in out.columns if c not in ["isolate", "protein"]]

    # First non-null *value* per isolate; if none, use literal "NA"
    def first_valid_or_NA(s):
        for x in s:
            if pd.notna(x):
                return x
        return "NA"

    per_iso_defaults = out.groupby("isolate")[cols_to_fill].agg(first_valid_or_NA)

    mask = out["protein"].eq(output_name)
    for c in cols_to_fill:
        filled = out.loc[mask, "isolate"].map(per_iso_defaults[c])
        # only replace true-missing; preserve existing literals including "NA"
        out.loc[mask, c] = out.loc[mask, c].where(~out.loc[mask, c].isna(), filled)

    # If anything is still a true-missing, make it literal "NA"
    out[cols_to_fill] = out[cols_to_fill].fillna("NA")

    # Correct the id column to reflect new protein names
    out['id'] = out[new_id_name].astype(str).agg('|'.join, axis=1)
    return out

# %%
## Save resulting dataframe 

# Save the dataframe as a CSV file
def save_fasta_as_dataframe(sequence, output_path):
    sequence.to_csv(output_path, index=False, quoting=1)
    print(f"Data saved to {output_path}")
    return

# %%
## Run the code

# Main function to read, process, and save fasta data
def fasta_read_combine(config_path: str | Path):
    # Read configuration
    cfg = ReadCombineConfig(config_path)

    # Read fasta file
    df = read_fasta_to_dataframe(cfg.fasta_path)

    # Process the dataframe by splitting and merging
    if cfg.column_names:
        df = split_by_id(df, cfg.column_names)
    df = merge_protein_sequences(
        sequence=df, 
        output_name=cfg.output_name,
        allowed=cfg.allowed,
        order=cfg.order,
        sequence_sort_order=cfg.sequence_sort_order,
        new_id_name=cfg.columns_for_new_id
    )

    # Save the resulting dataframe
    save_fasta_as_dataframe(df, cfg.output_path)
    return df

# %%
if __name__ == "__main__":
    fasta_read_combine()
