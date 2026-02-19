#!/usr/bin/env python

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open(os.path.join('/tiago_public_ws/src/tiago_data_generation/', 'config/datagen_controller.yaml')) as f:
    config = yaml.safe_load(f)

df = pd.read_csv(os.path.join('/home/anacg/.ros', config['datagen_controller']['filename']))
df['distance'] = np.sqrt(np.square(df['ee_x']) + np.square(df['ee_y']) + np.square(df['ee_z']))

vbles = ['ee_x', 'ee_y', 'ee_z', 'distance']

# Generated data
print(df[vbles].describe())

plt.figure(figsize=(20, 20))
colors = plt.cm.viridis(np.linspace(0, 1, len(vbles)))

for i, col in enumerate(vbles):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[col], color=colors[i])
    plt.title(col)
    plt.tight_layout()

plt.show()
plt.close()

# After standardization
df_std = (df - df.mean()) / df.std()

print(df_std[vbles].describe())

plt.figure(figsize=(20, 20))
colors = plt.cm.viridis(np.linspace(0, 1, len(vbles)))

for i, col in enumerate(vbles):
    plt.subplot(2, 3, i + 1)
    plt.hist(df_std[col], color=colors[i])
    plt.title(col)
    plt.tight_layout()

plt.show()
plt.close()

# After normalization
df_norm = (df - df.min()) / (df.max() - df.min())

print(df_norm[vbles].describe())

plt.figure(figsize=(20, 20))
colors = plt.cm.viridis(np.linspace(0, 1, len(vbles)))

for i, col in enumerate(vbles):
    plt.subplot(2, 3, i + 1)
    plt.hist(df_norm[col], color=colors[i])
    plt.title(col)
    plt.tight_layout()

plt.show()
plt.close()
