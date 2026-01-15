"""
Created on Mon Dec 1 15:37:21 2025

@author: rstutters
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("field_data\mapping_data\InnerCoilMSR3.csv", header=16)

calibrate = df.query('pos == "up" or pos == "down"')
df = pd.concat([df, calibrate]).drop_duplicates(keep=False)
df['pos'] = pd.to_numeric(df['pos'], errors='coerce')

# correct for fluxgate offset
up_x = calibrate.loc[calibrate['pos'] == 'up', 'Bx (V)'].mean()
down_x = calibrate.loc[calibrate['pos'] == 'down', 'Bx (V)'].mean()
offset_x = (up_x + down_x) / 2

up_z = calibrate.loc[calibrate['pos'] == 'up', 'Bz (V)'].mean()
down_z = calibrate.loc[calibrate['pos'] == 'down', 'Bz (V)'].mean()
offset_z = (up_z + down_z) / 2

offset_y = calibrate['By (V)'].mean()

df['Bx (V)'] -= offset_x
df['By (V)'] -= offset_y
df['Bz (V)'] -= offset_z

# flip x axis
df['pos'] = 120 - df['pos']

# scale fields
df['Bx (V)'] = df['Bx (V)'] * 100
df['By (V)'] = df['By (V)'] * 100
df['Bz (V)'] = df['Bz (V)'] * 100
df['dBx (V)'] = df['dBx (V)'] * 100
df['dBy (V)'] = df['dBy (V)'] * 100
df['dBz (V)'] = df['dBz (V)'] * 100
df = df.rename(columns={'Bx (V)': 'Bx (uT)', 'By (V)': 'By (uT)', 'Bz (V)': 'Bz (uT)', 'dBx (V)': 'dBx (uT)', 'dBy (V)': 'dBy (uT)', 'dBz (V)': 'dBz (uT)', 'pos': 'pos (cm)'})
df = df.drop(columns='orientation', axis=1)

# drop last 4 rows
# df = df[:-4]

# convert to MSR coordinates
df = df.rename(columns={'By (uT)': 'Bz (uT)', 'Bz (uT)': 'By (uT)', 'dBy (uT)': 'dBz (uT)', 'dBz (uT)': 'dBy (uT)'})

# export cleaned data
df.to_csv("field_data\mapping_data\cleanedInnerCoilMSR3.csv", index=False)

plt.errorbar(df['pos (cm)'], df['Bx (uT)'], xerr=0.2, yerr = df['dBx (uT)'], fmt='.', label='Bx', color='blue')
plt.errorbar(df['pos (cm)'], df['By (uT)'], xerr=0.2, yerr = df['dBy (uT)'], fmt='.', label='By', color='green')
plt.errorbar(df['pos (cm)'], df['Bz (uT)'], xerr=0.2, yerr = df['dBz (uT)'], fmt='.', label='Bz', color='orange')

plt.vlines([120-0, 120-7.5, 120-15, 120-17.5, 120-37.5, 120-62.5], -75, 50, label="MSR Layers", colors='green', linestyles='dotted')
plt.vlines([120-75.397], -75, 50, label="Coil Start", colors='blue', linestyles='dotted')

# Add labels, legend, and title
plt.xlim(125, -5)
plt.ylim(-55, 50)
plt.xlabel('Axial Position (y) (cm)', fontsize='16')
plt.ylabel('B (uT)', fontsize='16')
plt.grid()
plt.title('MSR South UCN Port Inner Coil Magnetic Fields (0.1A)', fontsize='16')
plt.legend()

# Show the plot
plt.show()