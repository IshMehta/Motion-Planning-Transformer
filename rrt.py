"""

Path planning with Rapidly-Exploring Random Trees (RRT)

author: Aakash(@nimrobotics)
web: nimrobotics.github.io

"""

import cv2
import numpy as np
import math
import random
import os
import time
from tqdm import tqdm
from os import path as osp
import pickle

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []

def any_less_than(threshold, numbers):
    return any(num < threshold for num in numbers)
# check collision
def collision(x1,y1,x2,y2):
    color=[]
    #print(x1, x2, y1, y2)
    x = list(np.arange(x1,x2,(x2-x1+1.e-5)/100))
    y = list(((y2-y1)/(x2-x1+1.e-5))*(x-x1+1.e-5) + y1)
    #print("collision",x,y)
    for i in range(len(x)):
        #print(int(x[i]),int(y[i]))
        color.append(img[int(y[i]),int(x[i])])
    #Change from the original code to all for collisions with grey boundary
    #Threshold of 150 was emperically chosen
    if any_less_than(150, color):
        return True #collision
    else:
        return False #no-collision

# check the  collision with obstacle and trim
def check_collision(x1,y1,x2,y2, goal):
    _,theta = dist_and_angle(x2,y2,x1,y1)
    x=x2 + stepSize*np.cos(theta)
    y=y2 + stepSize*np.sin(theta)
    #print(x2,y2,x1,y1)
    #print("theta",theta)
    #print("check_collision",x,y)

    # TODO: trim the branch if its going out of image area
    # print("Image shape",img.shape)
    hy,hx=img.shape
    if y<0 or y>hy or x<0 or x>hx:
        #print("Point out of image bound")
        directCon = False
        nodeCon = False
    else:
        # check direct connection
        if collision(x,y,goal[0],goal[1]):
            directCon = False
        else:
            directCon=True

        # check connection between two nodes
        if collision(x,y,x2,y2):
            nodeCon = False
        else:
            nodeCon = True

    return(x,y,directCon,nodeCon)

# return dist and angle b/w new point and nearest node
def dist_and_angle(x1,y1,x2,y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2-y1, x2-x1)
    return(dist,angle)

# return the neaerst node index
def nearest_node(x,y):
    temp_dist=[]
    for i in range(len(node_list)):
        dist,_ = dist_and_angle(x,y,node_list[i].x,node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))

# generate a random point in the image space
def rnd_point(h,l):
    new_y = random.randint(0, h)
    new_x = random.randint(0, l)
    return (new_x,new_y)


def RRT(img, img2, start, end, stepSize, debug):
    path = []
    path.append([start[0], start[1]])
    h,l= img.shape # dim of the loaded image
    # print(img.shape) # (384, 683)
    # print(h,l)

    # insert the starting point in the node class
    # node_list = [0] # list to store all the node points         
    node_list[0] = Nodes(start[0],start[1])
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])

    # display start and end
    if debug:
        cv2.circle(img2, (start[0],start[1]), 5,(0,0,255),thickness=3, lineType=8)
        cv2.circle(img2, (end[0],end[1]), 5,(0,0,255),thickness=3, lineType=8)

    i=1
    pathFound = False
    while pathFound==False:
        #print(node_list)
        nx,ny = rnd_point(h,l)
        if debug:
            print("Random points:",nx,ny)

        nearest_ind = nearest_node(nx,ny)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y
        if debug:
            print("Nearest node coordinates:",nearest_x,nearest_y)

        #check direct connection
        tx,ty,directCon,nodeCon = check_collision(nx,ny,nearest_x,nearest_y, goal_pos)
        if debug:
            print("Check collision:",tx,ty,directCon,nodeCon)

        if directCon and nodeCon:
            if debug:
                print("Node can connect directly with end")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            if debug:
                cv2.circle(img2, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
                cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
                cv2.line(img2, (int(tx),int(ty)), (end[0],end[1]), (255,0,0), thickness=2, lineType=8)
            if debug:
                print("Path has been found")
            
            #print("parent_x",node_list[i].parent_x)
            #print(start, end)
            for j in range(len(node_list[i].parent_x)-1):
                path.append([node_list[i].parent_x[j+1], node_list[i].parent_y[j+1]])
                if debug:
                    cv2.line(img2, (int(node_list[i].parent_x[j]),int(node_list[i].parent_y[j])), (int(node_list[i].parent_x[j+1]),int(node_list[i].parent_y[j+1])), (255,0,0), thickness=2, lineType=8)
            path.append([end[0], end[1]])
            path = [pix2geom(pos) for pos in path]
            #print(path)
            # cv2.waitKey(1)
            if debug:
                cv2.imwrite("media/"+str(i)+".jpg",img2)
                cv2.imwrite("out.jpg",img2)
            return np.flipud(np.array(path))
            

        elif nodeCon:
            if debug:
                print("Nodes connected")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            # print(i)
            # print(node_list[nearest_ind].parent_y)
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            i=i+1
            # display
            if debug:
                cv2.circle(img2, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
                cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
                cv2.imwrite("media/"+str(i)+".jpg",img2)
                cv2.imshow("sdc",img2)
                cv2.waitKey(1)
            continue

        else:
            #print("No direct con. and no node con. :( Generating new rnd numbers")
            continue
    

def draw_circle(event,x,y,flags,param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2,(x,y),5,(255,0,0),-1)
        coordinates.append(x)
        coordinates.append(y)

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


def createNewDataSet(env_string, img, list_of_pickle):
    """
    Mimicing the dataset ompl dataset
    """
    
    
      


if __name__ == '__main__':
    if not osp.exists("new_maze"):
        os.makedirs("new_maze/train")
        os.makedirs("new_maze/val")
    
    stepSize = 5
    #should be 3000
    for i in tqdm(range(1), desc = "Maze Training Progress"):
        #print(f'We are on maze {i} of {3000}')
        old_envFolder = f'maze4\\train\env{i:06d}'
        new_envFolder = f'new_maze\\train\env{i:06d}'
        map_file_path = osp.join(old_envFolder, f'map_{i}.png')
        new_map_file_path = osp.join(new_envFolder, f'map_{i}.png')
        os.makedirs(new_envFolder)
        img = cv2.imread(map_file_path, 0)
        img2 = cv2.imread(map_file_path)
        cv2.imwrite(new_map_file_path, img2)
        #should be 25
        for j in range(1):
            #print(f'We are on path {j} of {25} for maze {i}')
            start_pos, goal_pos = getStart_and_goal(old_envFolder, j)
            #print(start_pos, goal_pos)
            time_list = []
            number_of_times = 5
            rrt_path = []
            data = {}
            for k in range(number_of_times):
                success = False
                while success == False:
                    try:
                        node_list = [0]
                        start_time = time.time()
                        rrt_path = RRT(img, img2, start_pos, goal_pos, stepSize, True)
                        end_time = time.time()
                        print(f'Succeeded on map {i} at ({start_pos[0]}, {start_pos[1]}) , ({goal_pos[0]}, {goal_pos[1]}),  Iteration: {k+1}/{number_of_times}') 
                        success = True
                    except Exception as e:
                            print(f'Failed on map {i} at ({start_pos[0]}, {start_pos[1]}) , ({goal_pos[0]}, {goal_pos[1]}), Trying again, Iteration{k+1}/{number_of_times}')
                time_list.append(end_time - start_time)
            print(f'Average time for map {i} with start pos ({start_pos[0]}, {start_pos[1]}) and end pos ({goal_pos[0]}, {goal_pos[1]}) was {sum(time_list) / len(time_list)}')
            data['path'] = rrt_path
            data['times'] = time_list
            with open(osp.join(new_envFolder, f'path_{j}.p'), 'wb') as f:
                pickle.dump(data, f)
    #should be 500
        """ for i in tqdm(range(2), desc = "Maze Validation Progress"):
        #print(f'We are on maze {i} of {3000}')
        old_envFolder = f'maze4\\val\env{i:06d}'
        map_file_path = osp.join(old_envFolder, f'map_{i}.png')
        #should be 1
        for j in range(1):
            #print(f'We are on path {j} of {25} for maze {i}')
            start_pos, goal_pos = getStart_and_goal(old_envFolder, j)
            #print(start_pos, goal_pos)
            img = cv2.imread(map_file_path, 0)
            img2 = cv2.imread(map_file_path)
            time_list = []
            number_of_times = 5
            for k in range(number_of_times):
                success = False
                while success == False:
                    try:
                        node_list = [0]
                        start_time = time.time()
                        rrt_path = RRT(img, img2, start_pos, goal_pos, 10, False)
                        end_time = time.time()
                        print(f'Succeeded on map {i} at ({start_pos[0]}, {start_pos[1]}) , ({goal_pos[0]}, {goal_pos[1]}),  Iteration: {k+1}/{number_of_times}')
                        success = True
                    except Exception as e:
                            print(f'Failed on map {i} at ({start_pos[0]}, {start_pos[1]}) , ({goal_pos[0]}, {goal_pos[1]}), Trying again, Iteration{k+1}/{number_of_times}')
                
               
                time_list.append(end_time - start_time)
            print(f'Average time for map {i} with start pos ({start_pos[0]}, {start_pos[1]}) and end pos ({goal_pos[0]}, {goal_pos[1]}) was {sum(time_list) / len(time_list)}')
            
            #rrt_path = RRT(img, img2, start_pos, goal_pos, 15, debug=True)
            #print(rrt_path)
            """
