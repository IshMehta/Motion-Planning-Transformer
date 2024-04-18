import rrt
import os
from os import path as osp
import pickle 
import numpy as np
import cv2


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

#access the mazes from maze4 folder

    


#access the start and goal nodes for the algorithm to run
def getStart_and_goal(filepath, iteration):
    """
    Gets the start and goal location from the maze4 data set
    param filepath: The filepath where the .p files are located
    param iteration: The current iteration we are on (0-24)

    """
    data = pickle.load(open(osp.join(filepath, f'path_{iteration}.p'), 'rb'))
    path = data['path_interpolated']
    goal_pos = geom2pix(path[0, :])
    start_pos = geom2pix(path[-1, :])
    return (start_pos, goal_pos)
        



#run rrt algo and save path without creating the images while also recording the time it takes to run

if __name__ == '__main__':
    print(os.getcwd())
    #should be 3000
    for i in range(1):
        old_envFolder = f'maze4\\train\env{i:06d}'
        map_file_path = osp.join(old_envFolder, f'map_{i}.png')
        #should be 25
        for j in range(1):
            start_pos, goal_pos = getStart_and_goal(old_envFolder, j)
            img = cv2.imread(map_file_path, 0)
            img2 = cv2.imread(map_file_path)
            rrt_path = rrt.RRT(img, img2, start_pos, goal_pos, 10, debug=False)
            print(rrt_path)
            success = False
            while not success:
                try:
                    rrt_path = RRT(img, img2, start_pos, goal_pos, 10, debug=False)
                    print(rrt_path)
                    success = True
                except Exception as e:
                    print("Failed trying again")


