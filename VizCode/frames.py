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

frames = pd.DataFrame.from_records(zarr_dataset.frames, columns = ['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval', 'ego_translation','ego_rotation'])

frames[['ego_translation_x', 'ego_translation_y', 'ego_translation_z']] = frames['ego_translation'].to_list()
frames_new = frames.drop('ego_translation', axis=1)
frames_new = frames_new[['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval',
                         'ego_translation_x', 'ego_translation_y', 'ego_translation_z', 'ego_rotation']]

f = plt.figure(figsize=(16, 8))
gs = f.add_gridspec(1, 3)

with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,0])
    sns.distplot(frames_new['ego_translation_x'], color='Orange')
    plt.title('Ego Translation Distribution X')
    
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,1])
    sns.distplot(frames_new['ego_translation_y'], color='Red')
    plt.title('Ego Translation Distribution Y')
    
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,2])
    sns.distplot(frames_new['ego_translation_z'], color='Green')
    plt.title('Ego Translation Distribution Z')
    
f.tight_layout()

f = plt.figure(figsize=(16, 6))
gs = f.add_gridspec(1, 3)

with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0,0])
    plt.scatter(frames_new['ego_translation_x'], frames_new['ego_translation_y'],
                    color='darkkhaki', marker='+')
    plt.title('Ego Translation X-Y')
    plt.xlabel('ego_translation_x')
    plt.ylabel('ego_translation_y')
    
with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0,1])
    plt.scatter(frames_new['ego_translation_y'], frames_new['ego_translation_z'],
                    color='slateblue', marker='*')
    plt.title('Ego Translation Distribution Y-Z')
    plt.xlabel('ego_translation_y')
    plt.ylabel('ego_translation_z')
    
with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0,2])
    plt.scatter(frames_new['ego_translation_z'], frames_new['ego_translation_x'],
                    color='turquoise', marker='^')
    plt.title('Ego Translation Distribution Z-X')
    plt.xlabel('ego_translation_z')
    plt.ylabel('ego_translation_x')
    
f.tight_layout()

fig, ax = plt.subplots(3,3,figsize=(16,16))
colors = ['red', 'blue', 'green', 'magenta', 'orange', 'darkblue', 'black', 'cyan', 'darkgreen']
for i in range(0,3):
    for j in range(0,3):
        df = frames_new['ego_rotation'].apply(lambda x: x[i][j])
        plt.subplot(3,3,i * 3 + j + 1)
        sns.distplot(df, hist=False, color = colors[ i * 3 + j  ])
        plt.xlabel(f'r[ {i + 1} ][ {j + 1} ]')
fig.suptitle("Ego rotation angles distribution", size=14)
plt.show()