#from generateMazeMaps import generate_random_maze
import pickle
import os
from os import path as osp
import numpy as np
from skimage import io
import torch
import json
from transformer import Models
from eval_model import get_patch
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score


def pix2geom(pos, res=0.05, length=24):
    """
    Converts pixel co-ordinates to geometrical positions. 
    :param pos: The (x,y) pixel co-ordinates.
    :param res: The distance represented by each pixel.
    :param length: The length of the map in meters.
    :returns (float, float): The associated eucledian co-ordinates.
    """
    return (pos[0]*res, length-pos[1]*res)

def geom2pix(pos, res=0.05, size=(480, 480)):
    """
    Convert geometrical position to pixel co-ordinates. The origin 
    is assumed to be at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    NOTE: The Pixel co-ordinates are represented as follows:
    (0,0)------ X ----------->|
    |                         |  
    |                         |  
    |                         |  
    |                         |  
    Y                         |
    |                         |
    |                         |  
    v                         |  
    ---------------------------  
    """
    return (int(np.floor(pos[0]/res)), int(size[0]-1-np.floor(pos[1]/res)))


def load_trained_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    modelPath = 'checkpoint.pt'
    checkpoint = torch.load("checkpoint.pt")
    modelFile = 'training_results/model_params.json'
    model_param = json.load(open(modelFile))
    transformer = Models.Transformer(**model_param)
    _ = transformer.to(device)
    transformer.load_state_dict(checkpoint['model_state_dict']) 
    _ = transformer.eval()
    return transformer

def ground_truth(path, data, MazeMap):
    pathPix = [geom2pix(pos) for pos in path]

    uniqueId = []
    for pos in pathPix:
        if pos not in uniqueId:
            uniqueId.append(pos)
    pathPixelPos = np.array([geom2pix(pos) for pos in data['path']])# Generate Patch Maps
    truePatchMap = np.zeros_like(MazeMap)
    map_size = MazeMap.shape
    receptive_field = 32
    for pos in uniqueId:
        goal_start_x = max(0, pos[0]- receptive_field//2)
        goal_start_y = max(0, pos[1]- receptive_field//2)
        goal_end_x = min(map_size[0], pos[0]+ receptive_field//2)
        goal_end_y = min(map_size[1], pos[1]+ receptive_field//2)
        truePatchMap[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
    return truePatchMap

def gen_metrics (truePatchMap, pred_pathMap):
    m = []

    m.append(accuracy_score(truePatchMap.flatten(), pred_pathMap.flatten()))
    m.append(precision_score(truePatchMap.flatten(), pred_pathMap.flatten()))
    m.append(recall_score(truePatchMap.flatten(), pred_pathMap.flatten()))
    #print(metrics)
    return m


if __name__ == "__main__":
    transformer = load_trained_model()
    metrics  = []
    for i in tqdm(range(100), desc = "Maze Evaluation Progres"):
        val_envFolder = f'new_maze/val/env{i:06d}'
        map_file_path = osp.join(val_envFolder, f'map_{i}.png')
        data = pickle.load(open(osp.join(val_envFolder, f'path_{0}.p'), 'rb'))
        path = data['path_interpolated']
        img = io.imread(osp.join(val_envFolder, f'map_{i}.png'), as_gray=True)
        
        goal_pos = geom2pix(path[0, :])
        start_pos = geom2pix(path[-1, :])
        

        truePatchMap = ground_truth(path, data, img)
        patch_map, _ = get_patch(transformer, start_pos, goal_pos, img)

        metrics.append(gen_metrics(truePatchMap, patch_map))
    metrics = np.array(metrics)
    print(np.mean(metrics, axis = 0))
        


        
