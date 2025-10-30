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
from sklearn.preprocessing import MinMaxScaler

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
        self.cmap_name = p["cmap_name"]
        self.invert = p["invert"]

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
def generate_palette_rgb_v5(
            cmap, 
            data_minmax, 
            model_number,
            invert: str
            ):      
      color_list = []

      # Calculate the color mapping
      for idx, value in enumerate(data_minmax):
            # Correct if the colormap is inverted
            if invert == "True":
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
## Scale and normalize the data
def data_scaler(entropy_mean_values):
    data_minmax = MinMaxScaler().fit_transform(entropy_mean_values.values.reshape(-1, 1)).flatten()
    return data_minmax

# %%
## Generate the color map
def generate_color_map(config_path: str | Path):
    # Load configuration from YAML file
    cfg = ColorConfig(config_path)

    # Extract the file name without extension from the input path
    polymerase_analysis_file_name_no_ext = Path(cfg.input_path).stem
    
    # Read entropy mean values from CSV file
    polymerase_entropy_mean = pd.read_csv(
        cfg.input_path
        )

    # Construct colormap name, appending "_invert" if inversion is enabled
    cmap_name_file = cfg.cmap_name
    if cfg.invert == "True":
        cmap_name_file = cmap_name_file + "_invert"
    
    # Generate output file name
    cxc_file_name = f"entropy_color_mapping_{cmap_name_file}"
    color_list_total = []
    
    # Load the colormap from seaborn
    cmap = sns.color_palette(cfg.cmap_name, as_cmap=True)

    # Normalize entropy values to 0-1 range using MinMaxScaler
    data_minmax = data_scaler(polymerase_entropy_mean)

    # Generate RGB color mappings for the normalized data
    model_number = 1
    color_list = generate_palette_rgb_v5(
            cmap=cmap, 
            data_minmax=data_minmax, 
            model_number=model_number,
            invert=cfg.invert
            )
    color_list_total.extend(color_list)

    # Write color mappings to output CXC file
    output_file = cfg.output_dir / f"{cxc_file_name}_{polymerase_analysis_file_name_no_ext}.cxc"
    with open(output_file, "w") as f:
        for residue in color_list_total:
            f.write(f"{residue}\n")

    return

# %%
if __name__ == "__main__":
    generate_color_map()
