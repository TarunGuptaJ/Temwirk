# import packages
import os, gc
import zarr
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from typing import Dict
from collections import Counter
from prettytable import PrettyTable

#level5 toolkit
from l5kit.data import PERCEPTION_LABELS
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# level5 toolkit 
from l5kit.configs import load_config_data
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, draw_reference_trajectory, TARGET_POINTS_COLOR
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from colorama import Fore, Back, Style


train = zarr.open("Dataset/scenes/train.zarr")
validation = zarr.open("Dataset/scenes/validate.zarr")
test = zarr.open("Dataset/scenes/test.zarr/")
train.info

dm = LocalDataManager()
dataset_path = dm.require(cfg["val_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

agents = pd.DataFrame.from_records(zarr_dataset.agents, columns = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities'])

agents[['centroid_x','centroid_y']] = agents['centroid'].to_list()
agents = agents.drop('centroid', axis=1)
agents_new = agents[["centroid_x", "centroid_y", "extent", "yaw", "velocity", "track_id", "label_probabilities"]]
del agents

fig, ax = plt.subplots(1,1,figsize=(8,8))
plt.scatter(agents_new['centroid_x'], agents_new['centroid_y'], marker='+')
plt.xlabel('x', fontsize=11); plt.ylabel('y', fontsize=11)
plt.title("Centroids distribution (sample.zarr)")
plt.show()

agents_new[['extent_x','extent_y', 'extent_z']] = agents_new['extent'].to_list()
agents_new = agents_new.drop('extent', axis=1)
agents = agents_new[["centroid_x", "centroid_y", 'extent_x', 'extent_y', 'extent_z', "yaw", "velocity", "track_id", "label_probabilities"]]
del agents_new

sns.axes_style("white")

fig, ax = plt.subplots(1,3,figsize=(16,5))

plt.subplot(1,3,1)
sns.kdeplot(agents['extent_x'], shade=True, color='red');
plt.title("Extent_x distribution")

plt.subplot(1,3,2)
sns.kdeplot(agents['extent_y'], shade=True, color='steelblue');
plt.title("Extent_y distribution")

plt.subplot(1,3,3)
sns.kdeplot(agents['extent_z'], shade=True, color='green');
plt.title("Extent_z distribution")

plt.show()

sns.set_style('whitegrid')

fig, ax = plt.subplots(1,3,figsize=(16,5))
plt.subplot(1,3,1)
plt.scatter(agents['extent_x'], agents['extent_y'], marker='*')
plt.xlabel('ex', fontsize=11); plt.ylabel('ey', fontsize=11)
plt.title("Extent: ex-ey")

plt.subplot(1,3,2)
plt.scatter(agents['extent_y'], agents['extent_z'], marker='*', color="red")
plt.xlabel('ey', fontsize=11); plt.ylabel('ez', fontsize=11)
plt.title("Extent: ey-ez")

plt.subplot(1,3,3)
plt.scatter(agents['extent_z'], agents['extent_x'], marker='*', color="green")
plt.xlabel('ez', fontsize=11); plt.ylabel('ex', fontsize=11)
plt.title("Extent: ez-ex")

plt.show()

fig, ax = plt.subplots(1,1,figsize=(10,8))
sns.distplot(agents['yaw'])
plt.title("Yaw Distribution")
plt.show()

agents[['velocity_x','velocity_y']] = agents['velocity'].to_list()
agents_vel = agents.drop('velocity', axis=1)
agents_v = agents_vel[["centroid_x", "centroid_y", 'extent_x', 'extent_y', 'extent_z', "yaw", "velocity_x", "velocity_y", "track_id", "label_probabilities"]]
del agents
agents_v

fig, ax = plt.subplots(1,1,figsize=(10,8))

with sns.axes_style("whitegrid"):
    sns.scatterplot(x=agents_v["velocity_x"], y=agents_v["velocity_y"], color='k');
    plt.title('Velocity Distribution')

agents = zarr_dataset.agents
probabilities = agents["label_probabilities"]
labels_indexes = np.argmax(probabilities, axis=1)
counts = []
for idx_label, label in enumerate(PERCEPTION_LABELS):
    counts.append(np.sum(labels_indexes == idx_label))
    
table = PrettyTable(field_names=["label", "counts"])
for count, label in zip(counts, PERCEPTION_LABELS):
    table.add_row([label, count])
print(table)