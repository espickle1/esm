# %%
## Import necessary libraries
from __future__ import annotations

# Standard libraries
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Any

# Specialized libraries
import seaborn as sns
from sklearn.preprocessing import (
    MinMaxScaler, 
    QuantileTransformer, 
    PowerTransformer, 
    StandardScaler, 
    RobustScaler
)

# %%
## Create class containing read and combine configuration
class ColorConfig:
    def __init__(self, config_path: str | Path):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        # Loading paths
        self.input_dir = Path(self._cfg["paths"]["input_dir"])
        self.output_dir = Path(self._cfg["paths"]["output_dir"])
        self.input_path = self.input_dir / self._cfg["input"]["input_file_name"]

        # Inference parameters
        p = self._cfg["parameters"]
        self.transform_method = p["transform_method"]
        # Color mapping
        self.color_mapping = p["color_mapping"]
        self.cmap_name = p["cmap_name"]
        self.color_invert = p["color_mapping_invert"]
        # Transparency mapping
        self.transparency_mapping = p["transparency_mapping"]
        self.transparency_invert = p["transparency_mapping_invert"]

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
## Calculate the rgb color based on the input scalar
def generate_palette_rgb(
            cmap, 
            data_minmax, 
            model_number,
            color_invert: str
            ):      
      color_list = []

      # Calculate the color mapping
      for idx, value in enumerate(data_minmax):
            # Correct if the colormap is inverted
            if color_invert == "True":
                rgba_0_1 = cmap(1 - value)
            else:
                rgba_0_1 = cmap(value)

            # Calculate the RGB color mapping
            rgb_255 = tuple(int(round(c * 255)) for c in rgba_0_1)
            rgb_255 = [max(color_value, 0) for color_value in rgb_255]
            color_text = f"color #{model_number}:{idx + 1} {rgb_255[0]},{rgb_255[1]},{rgb_255[2]},{rgb_255[3]} atoms,cartoons,surface"
            color_list.append(str(color_text))
            
      return color_list

# %%
## Calculate transparency based on the input scalar
def generate_transparency(
            data_minmax, 
            model_number,
            transparency_invert: str
            ):      
      transparency_list = []

      # Calculate the transparency mapping
      for idx, value in enumerate(data_minmax):
            # Correct if the transparency is inverted
            if transparency_invert == "True":
                transparency = 1 - value
            else:
                transparency = value

            # Calculate the transparency mapping
            transparency = max(0, min(1, transparency))
            transparency_text = f"transparency #{model_number}:{idx + 1} {transparency} atoms,cartoons,surface"
            transparency_list.append(str(transparency_text))

      return transparency_list

# %%
## Scale and normalize the data
def data_scaler(entropy_mean_values, transform_method: str):
    """
    Apply various scaling/transformation methods to entropy values.
    
    Args:
        entropy_mean_values: Input data to be scaled
        transform_method: Type of transformation ('quantile', 'power', 'standard', 'robust', or other)
    
    Returns:
        Scaled/transformed values as a flattened numpy array
    """
    if transform_method == "quantile":
        # Quantile transformer maps values to uniform distribution [0, 1]
        scaled_values = QuantileTransformer(output_distribution="uniform").fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    elif transform_method == "power":
        # Power transformer applies Yeo-Johnson transformation for skewed data
        scaled_values = PowerTransformer(method="yeo-johnson").fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    elif transform_method == "standard":
        # Standard scaler normalizes to zero mean and unit variance
        scaled_values = StandardScaler().fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    elif transform_method == "robust":
        # Robust scaler uses median and IQR, less sensitive to outliers
        scaled_values = RobustScaler().fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    else:
        # Default: return values as-is without transformation
        scaled_values = entropy_mean_values.values.flatten()
        
    return scaled_values

# %%
## Generate the color map
def generate_color_map(config_path: str):
    """
    Generate color and transparency mappings for molecular visualization.
    
    Args:
        config_path: Path to the YAML configuration file (default: "config.yaml")
    """
    # Load configuration from YAML file
    cfg = ColorConfig(config_path)

    # Extract the file name without extension from the input path
    polymerase_analysis_file_name_no_ext = Path(cfg.input_path).stem
    
    # Read entropy mean values from CSV file
    polymerase_entropy_mean = pd.read_csv(cfg.input_path)

    # Construct colormap name, appending "_invert" if inversion is enabled
    cmap_name_file = cfg.cmap_name
    if cfg.color_invert == "True":
        cmap_name_file = cmap_name_file + "_invert"
    
    # Generate output file name
    cxc_file_name = f"entropy_color_mapping_{cmap_name_file}"
    if cfg.transparency_mapping == "True":
        cxc_file_name = cxc_file_name + "_transparency"

    # Initialize lists to hold color and transparency mappings for all residues
    color_list_total = []
    transparency_list_total = []
    
    # Load the colormap from seaborn
    cmap = sns.color_palette(cfg.cmap_name, as_cmap=True)

    # Scale and normalize entropy values with transformer
    polymerase_entropy_mean_transformed = data_scaler(
        polymerase_entropy_mean, 
        cfg.transform_method
        )

    # Normalize entropy values to 0-1 range using MinMaxScaler
    data_minmax = MinMaxScaler().fit_transform(
        polymerase_entropy_mean_transformed.reshape(-1, 1)).flatten()

    # Generate color and transparency mappings for the normalized data
    model_number = 1

    # Generate color mappings if enabled
    if cfg.color_mapping == "True":
        color_list = generate_palette_rgb(
            cmap=cmap, 
            data_minmax=data_minmax, 
            model_number=model_number,
            color_invert=cfg.color_invert
        )
        color_list_total.extend(color_list)
    
    # Generate transparency mappings if enabled
    if cfg.transparency_mapping == "True":
        transparency_list = generate_transparency(
            data_minmax=data_minmax, 
            model_number=model_number,
            transparency_invert=cfg.transparency_invert
        )
        transparency_list_total.extend(transparency_list)

    # Write color and transparency mappings to output CXC file
    output_file = cfg.output_dir / f"{cxc_file_name}_{polymerase_analysis_file_name_no_ext}.cxc"
    with open(output_file, "w") as f:
        if cfg.color_mapping == "True":
            for residue in color_list_total:
                f.write(f"{residue}\n")
        if cfg.transparency_mapping == "True":
            for residue in transparency_list_total:
                f.write(f"{residue}\n")
        else:
            f.write("select clear")

    return

# %%
if __name__ == "__main__":
    generate_color_map()
