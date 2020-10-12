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

scenes = pd.DataFrame.from_records(zarr_dataset.scenes, columns = ['frame_index_interval', 'host', 'start_time', 'end_time'])
scenes[['frame_start_index','frame_end_index']] = scenes['frame_index_interval'].to_list()
scenes_new = scenes.drop('frame_index_interval', axis=1)
scenes_new = scenes_new[["frame_start_index", "frame_end_index", 'host', 'start_time', 'end_time']]

f = plt.figure(figsize=(10, 8))
gs = f.add_gridspec(1, 2)

with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,0])
    sns.scatterplot(scenes_new['frame_start_index'], scenes_new['frame_end_index'])
    plt.title('Frame Index Interval Distribution')
    
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,1])
    sns.scatterplot(scenes_new['frame_start_index'], scenes_new['frame_end_index'], hue=scenes_new['host'])
    plt.title('Frame Index Interval Distribution (Grouped per host)')
    
f.tight_layout()

f = plt.figure(figsize=(10, 8))

with sns.axes_style("white"):
    sns.countplot(scenes_new['host']);
    plt.title("Host Count")