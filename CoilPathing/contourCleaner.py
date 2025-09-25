"""
Cleans exported COMSOL txt files containing 
contour points so they can be imported to SolidWorks

Code by Rylan Stutters

"""

import pandas as pd
import re
from io import StringIO

input_file = "\TestCoil-C1.txt"
path = "CoilPathing\Contours_in" + input_file


points = []
edges = []
with open(path, "r") as f:
    for line in f:
        # Remove whitespace
        stripped = line.strip()
        if not stripped:
            continue

        # Check if line looks numeric and has 2 or 3 entries
        if re.match(r"^[\d\s,.\-eE]+$", stripped):
            elements = stripped.split()
            if len(elements) == 3:
                points.append(stripped)
            elif len(elements) == 2:
                edges.append(stripped)


# Write numeric lines to a temporary string and convert to dataframe
point_str = "\n".join(points)
edge_str = "\n".join(edges)
df_p = pd.read_csv(StringIO(point_str), delim_whitespace=True, header=None)
df_e = pd.read_csv(StringIO(edge_str), delim_whitespace=True, header=None)

df_p = df_p.mul(1000)

point_order = df_e.iloc[:,0] - 1
df_p = df_p.reindex(point_order)

# Save cleaned output
df_p.to_csv("CoilPathing\Contours_out" + input_file, index=False, sep=",", float_format="%f", header=False)
