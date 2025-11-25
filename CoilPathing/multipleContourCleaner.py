"""
Cleans exported COMSOL txt files containing
contour points for several contours so they can 
be individually imported to SolidWorks

Code by Rylan Stutters

"""

import pandas as pd
import re
from io import StringIO
import os

input_file = "taperV2outer"
path = "CoilPathing\\Contours_in\\" + input_file + ".txt"


points = []
edges = []
iso = []
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
            elif len(elements) == 1:
                iso.append(stripped)


# Write numeric lines to a temporary string and convert to dataframe
point_str = "\n".join(points)
edge_str = "\n".join(edges)
iso_str = "\n".join(iso)
df_p = pd.read_csv(StringIO(point_str), delim_whitespace=True, header=None)
df_e = pd.read_csv(StringIO(edge_str), delim_whitespace=True, header=None)
df_i = pd.read_csv(StringIO(iso_str), delim_whitespace=True, header=None)

df_p = df_p.mul(1000)

# reorder points
point_order = df_e.iloc[:,0] - 1
df_p = df_p.reindex(point_order)

# seperate isolevels
isolevels = df_i[0].drop_duplicates()
df_p["iso"] = df_i
contours = []
for level in isolevels:
    df = df_p[df_p['iso'] == level]
    df = df.drop('iso', axis=1)
    contours.append(df)

# close contours
for i in range(len(contours)):
    closing_point = contours[i].iloc[0]
    contours[i] = pd.concat([contours[i], closing_point.to_frame().T], ignore_index=True)

# Save cleaned contours
os.mkdir("CoilPathing\\Contours_out\\" + input_file)
i = 0
for contour in contours:
    contour.to_csv("CoilPathing\\Contours_out\\" + input_file + "\\" + input_file + f"-C-{i}", index=False, sep=",", float_format="%f", header=False)
    i += 1
