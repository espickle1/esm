# %%
## Import dependencies
# Standard libraries
import pandas as pd

# Biology and structure libraries
from biotite.database import rcsb
import py3Dmol

# Visualization configuration loading
import yaml
from pathlib import Path
from typing import Any


# ESM3 and ESMC libraries
from esm.utils.structure.protein_chain import ProteinChain
from esm.sdk.api import ESMProtein

# %%
# Create class containing read and combine configuration
class VisualizationConfig:
    def __init__(self, config_path: str | Path):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)
        
        # Loading parameters
        p = self._cfg["parameters"]
        self.pdb_source = p["pdb_source"]
        self.pdb_id = p["pdb_id"]
        self.pdb_path = self.base_dir / self._cfg["input"]["pdb_file_path"]

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
## Convert the pdb to ESMProtein 
def pull_and_convert_pdb(pdb_path: str, pdb_id: str) -> ESMProtein:
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

# %%
## Visualization class
class ProteinVisualizer:
    @staticmethod
    # Visualize a pdb string
    def visualize_pdb(pdb_string, residue_colors=None):
        """
        Visualize a protein structure with optional custom residue coloring.
        
        Args:
            pdb_string: PDB format string
            residue_colors: Dict mapping residue indices to colors (e.g., {0: 'red', 1: 'blue'})
                          or list of colors for each residue
        """
        view = py3Dmol.view(width=400, height=400)
        view.addModel(pdb_string, "pdb")
        
        if residue_colors is None:
            # Default spectrum coloring
            view.setStyle({"cartoon": {"color": "spectrum"}})
        else:
            # Apply custom colors per residue
            if isinstance(residue_colors, dict):
                for resi, color in residue_colors.items():
                    view.setStyle({"resi": resi}, {"cartoon": {"color": color}})
            elif isinstance(residue_colors, list):
                for resi, color in enumerate(residue_colors):
                    view.setStyle({"resi": resi}, {"cartoon": {"color": color}})
        
        view.zoomTo()
        view.render()
        view.center()
        return view

    @staticmethod
    # Or visualize coordinates directly
    def visualize_3D_coordinates(coordinates, residue_colors=None):
        """
        This uses all Alanines
        """
        protein_with_same_coords = ESMProtein(coordinates=coordinates)
        pdb_string = protein_with_same_coords.to_pdb_string()
        return ProteinVisualizer.visualize_pdb(pdb_string, residue_colors)

    @staticmethod
    # Or visualize the ESMProtein object directly
    def visualize_3D_protein(protein, residue_colors=None):
        pdb_string = protein.to_pdb_string()
        return ProteinVisualizer.visualize_pdb(pdb_string, residue_colors)
    
    @staticmethod
    def color_by_values(values, colormap='viridis'):
        """
        Helper function to map numerical values to colors.
        
        Args:
            values: List or array of numerical values (one per residue)
            colormap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'coolwarm')
        
        Returns:
            List of hex color strings
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        values = np.array(values)
        normalized = (values - values.min()) / (values.max() - values.min())
        cmap = plt.cm.get_cmap(colormap)
        colors = [plt.matplotlib.colors.rgb2hex(cmap(val)[:3]) for val in normalized]
        return colors

# %%
## Retrieve and convert pdb file into ESMProtein object

cfg = VisualizationConfig(config_path="config/visualize_structure.yaml")

if cfg.pdb_source == "pdb_id":
    pdb_id = cfg.pdb_id
    structure = pull_and_convert_pdb(pdb_path=None, pdb_id=pdb_id)
elif cfg.pdb_source == "pdb_path":
    pdb_path = cfg.pdb_path
    structure = pull_and_convert_pdb(pdb_path=pdb_path, pdb_id=None)
else:
    raise ValueError("Invalid pdb_source in configuration. Must be either rcsb id or pdb file path.")

ProteinVisualizer.visualize_3D_protein(structure)

## Examples of custom coloring
## Example 1: Custom colors by residue index (dict)
residue_colors = {0: 'red', 5: 'blue', 10: 'green'}
ProteinVisualizer.visualize_3D_protein(structure, residue_colors=residue_colors)

## Example 2: Color all residues with a list
colors = ['red'] * 10 + ['blue'] * 10 + ['green'] * 10  # First 10 red, next 10 blue, etc.
ProteinVisualizer.visualize_3D_protein(structure, residue_colors=colors)

## Example 3: Color by numerical values (e.g., B-factors, predicted scores, etc.)
# Generate some example values (replace with your actual values)
import numpy as np
values = np.random.rand(len(structure.sequence))  # Random values for each residue
colors = ProteinVisualizer.color_by_values(values, colormap='coolwarm')
ProteinVisualizer.visualize_3D_protein(structure, residue_colors=colors)
