# The core code was taken from https://github.com/blazejosinski/lm_nav repository and then modified to our need

import base64
import json
import networkx
import numpy as np
import pickle
from string import Template
from IPython.display import display, Javascript, HTML

import lm_nav
from lm_nav.navigation_graph import NavigationGraph
from lm_nav import optimal_route, pipeline
import gdown
from tqdm import tqdm

use_large_graph = True
test = False

if use_large_graph and test:
  graph_file_gdrive_id = "186_WuE3caY0ADoaJrGurs9PACLsgSjup"
  all_routes = [
      (77, "All landmarks"),
      (108, "Take a right next to a stop sign. Look for a glass building, after passing by a white car."),
      (247, "Follow the road and take the right, you should see a blue semi-truck. Behind the truck, take a right next to an orange traffic cone. Go towards a blue dumpster and take left. Look for a picnic bench."),
  ]
  all_routes_gt = [
      [108, 210, 217, 220],
      [247, 254, 264, 275],
  ]
  landmarks_cache = eval("[['a glass building', 'a square with a large tree', 'a square with a tree', 'a white building', 'traffic lights', 'a white car', 'a disabled Parking spot', 'a trailer', 'a building with a red-black wall', 'a fire hydrant', 'a stop sign', 'an orange traffic cone', 'a manhole cover', 'a blue semi-truck', 'a red building', 'a picnic bench', 'a white truck', 'a white trailer', 'a traffic cone', 'a grove', 'a blue dumpster'],['a stop sign', 'a white car', 'a glass building'], ['a blue semi-truck', 'an orange traffic cone', 'a blue dumpster', 'a picnic bench'] ]")
elif use_large_graph:
    graph_file_gdrive_id = "186_WuE3caY0ADoaJrGurs9PACLsgSjup"
    all_routes = [
        (5, "Go straight toward the white building. Continue straight passing by a white truck until you reach a stop sign."),
        (5, "After passing a white building, take right next to a white truck. Then take left and go towards a square with a large tree. Go further, until you find a stop sign."),
        (173, "Start going around a building with a red-black wall and pass by a fire hydrant. Take a right and enter a grove. Continue straight and take a right, when you see a manhole cover. Go forward and left, and look for a trailer."),
        (108, "Take a right next to a stop sign. Look for a glass building, after passing by a white car."),
        (247, "Follow the road and take the right, you should see a blue semi-truck. Behind the truck, take a right next to an orange traffic cone. Go towards a blue dumpster and take left. Look for a picnic bench."),
        (70, "Go towards a white trailer. Then take left and go to the traffic lights. Take left again, and look for a traffic cone."),
        (215, "Go straight, passing by a stop sign and a  manhole cover. Next, you will see a disabled Parking spot and a red building."),
        (103, "First, you need to find a stop sign. Then take left and right and continue until you reach a square with a tree. Continue first straight, then right, until you find a white truck. The final destination is a white building."),
        (103, "Go to a stop sign. Continue straight, passing by a white truck. The final destination is a white building."),
        (211, "Go straight, until you find a glass building. Drive to a white car nearby. Drive to reach a stop sign, this is your destination.")
    ]
    # all_routes_gt = [
    #     [5, 8, 77],
    #     [5, 8, 23, 261, 77],
    #     [173, 160, 150, 191, 129, 45],
    #     [108, 210, 217, 220],
    #     [247, 254, 264, 275],
    #     [70, 39, 34, 257],
    #     [215, 194, 184, 170],
    #     [103, 267, 22, 8],
    #     [103, 16, 8],
    #     [211, 220, 217, 204],
    # ]
    all_routes_gt = [
        [5, 4, 69, 80,],
        [5,  4, 69, 68, 80,],
        [173, 177, 181, 184, 138, 80,],
        [108, 108, 78, 80,],
        [247, 45, 71, 0, 80,],
        [70, 70, 69, 80,],
        [215, 111, 109, 108, 80,],
        [103, 77, 16, 20, 80,],
        [103, 77, 20, 80,],
        [211,  108, 78, 80,]
    ]
    landmarks_cache = eval("[['a white building', 'a white truck', 'a stop sign'], ['a white building', 'a white truck', 'a square with a large tree', 'a stop sign'], ['a building with a red-black wall', 'a fire hydrant', 'a grove', 'a manhole cover', 'a trailer'], ['a stop sign', 'a white car', 'a glass building'], ['a blue semi-truck', 'an orange traffic cone', 'a blue dumpster', 'a picnic bench'], ['a white trailer', 'traffic lights', 'a traffic cone'], ['a stop sign', 'a manhole cover', 'a disabled Parking spot', 'a red building'], ['a stop sign', 'a square with a tree', 'a white truck', 'a white building'], ['a stop sign', 'a white truck', 'a white building'], ['a glass building', 'a white car', 'a stop sign']]")
else:
    graph_file_gdrive_id = "1SJiZmFDLcCnBGz2XuOLPewR1hz8ScwHi"
    all_routes = [
        (180, "Go straight towards a stop sign, take left and go until you reach a traffic cone. Take another left and then right going towards a blue box. From there take left and look for a baby stroller."),
        (215, "Go towards the blue box, take right and left until you reach a traffic cone. Take left and pass by a semi-truck until you find a big log."),
        (63, "Start at a traffic cone. Go towards a cardboard box and a parking lot. Continue driving, take a right, and pass by a picnic table. Take left and look for a stop sign."),
        (160, "Take first right towards a picnic table. Next, go to a square with a large tree, and take the left to another picnic table. Keep going until you reach a parking lot."),
        (61, "Go straight and take right next to a traffic cone. Go straight until you reach a parking lot. Take left, go through a lawn and look for a blue box."),
        (219, "Pass by a blue box and look for a big log. Take right and keep going straight, passing by a traffic cone. Take a right and finish at the parking lot."),
        (186, "Look for a traffic cone, take left and go straight until you find a square with a tree. Turn right, pass by a cardboard box and go to a picnic table."),
        (75, "Go straight pass a picnic table until you reach a street. Take right, pass by an orange trailer and take next right at an intersection. Next, take a right next to a traffic cone, take the next left, and pass by a baby stroller. Go straight and you will reach a parking lot."),
        (194, "Take a left when you see a traffic cone. Go straight passing by a semi-track and take left after you see a big log. Drive to a blue box and continue straight until you find a cardboard box next to a parking lot."),
        (133, "Take right at a traffic cone, and go straight until you reach a square with a big tree. Take right next and go straight until you find a baby stroller. Take left and right and look for an intersection."),
    ]
    # all_routes_gt = [
    #     [180,188, 224,220, 216],
    #     [215, 220, 226, 194, 134, 131],
    #     [63,75,78,149,157,165],
    #     [160,157, 149,202,38,45,50],
    #     [61, 78, 121],
    #     [219, 131, 182],
    #     [186, 15, 205, 44],
    #     [75, 52, 62, 69, 216, 240],
    #     [194, 134, 131, 220, 240],
    #     [133,138,230,216,63],
    # ]

    all_routes_gt = [
        [180, 184, 186, 189, 80,],
        [215, 207, 206, 205, 80,],
        [63, 65, 70, 71, 74, 80,],
        [160, 155, 154, 29, 80,],
        [61, 65, 66, 71, 80,],
        [219, 210, 207, 206, 80,],
        [186, 11, 1, 82, 80,],
        [75, 75, 36, 38, 78, 79, 80, 80,], # issue
        [194, 194, 4, 142, 2, 82, 80,], # issue
        [133, 12, 1, 82, 80,]
    ]
    landmarks_cache = eval("[['a stop sign', 'a traffic cone', 'a blue box', 'a baby stroller'], ['a blue box', 'a traffic cone', 'a semi-truck', 'a big log'], ['a traffic cone', 'a cardboard box', 'a parking lot', 'a picnic table', 'a stop sign'], ['a picnic table', 'a square with a large tree', 'another picnic table', 'a parking lot'], ['a traffic cone', 'a parking lot', 'a lawn', 'a blue box'], ['a blue box', 'a big log', 'a traffic cone', 'a parking lot'], ['a traffic cone', 'a square with a tree', 'a cardboard box', 'a picnic table'], ['a picnic table', 'a street', 'an orange trailer', 'an intersection', 'a traffic cone', 'a baby stroller', 'a parking lot'], ['a traffic cone', 'a semi-track', 'a big log', 'a blue box', 'a cardboard box', 'a parking lot'], ['a traffic cone', 'a square with a big tree', 'a baby stroller', 'an intersection']]")



alpha = 0.002
for i in range(5):
    alpha *= 0.1
    all_results = []
    graph_type = "large" if use_large_graph else "small"
    number_of_paths = 10
    for i in tqdm(range(number_of_paths)):
        graph_path = f"experiments/{graph_type}_graphs/{graph_type}_inx{i}_to_node80.pkl"
        graph = NavigationGraph(graph_path)
        start_idx = i
        load_idx = i+1 
        all_results_cur = [pipeline.full_pipeline(graph, start_node=start, instructions=description, alpha=alpha) if cached_landmarks is None else pipeline.full_pipeline(graph, start_node=start, landmarks=cached_landmarks, alpha=alpha) for ((start, description), cached_landmarks) in zip(all_routes[start_idx:load_idx], landmarks_cache[start_idx:load_idx])]
        all_results = all_results + all_results_cur


    matching_rate = 0
    total_acc = 0
    arrival_success = 0
    for inx in range(number_of_paths):
        gt_nodes = all_routes_gt[inx][1:]
        lm_nav_path = [tup[0] for tup in all_results[inx]["walk"] if tup[1] == -1]
        same_count = sum([1 for i, j in zip(gt_nodes, lm_nav_path) if i == j])
        accuracy = same_count / len(gt_nodes)
        total_acc += accuracy
        if gt_nodes[-1] == lm_nav_path[-1]:
            arrival_success += 1
    print('Alpha: ', alpha)
    print('Total matching rate: ', total_acc/number_of_paths)
    print('Arrival success: ', arrival_success/number_of_paths)
        
    descriptions_with_walks = [(route_input[1], [a[0] for a in route_output["walk"]])for route_input, route_output in zip(all_routes, all_results)]

    # Measure route distances and efficiency
    # Floyd-Warschal algorithm

    distance = np.zeros((graph.vert_count,graph.vert_count))
    distance.fill(1e9)

    for i in range(graph.vert_count):
        distance[i,i] = 0

    for u,v in graph._graph.edges():
        d = np.linalg.norm(graph._pos[u] - graph._pos[v])
        distance[u, v] = d
        distance[v, u] = d

    for k in range(graph.vert_count):
        for i in range(graph.vert_count):
            for j in range(graph.vert_count):
                if distance[i,j] > distance[i,k] + distance[k,j]:
                    distance[i,j] = distance[i,k] + distance[k,j]

    def path_length(path, distance):
        prev = None
        res = 0.
        for i in path:
            if prev is not None and i != prev:
                res += distance[prev,i]
            prev = i
        return res


    inx = 0
    path_length(descriptions_with_walks[inx][1], distance)

    walk_with_data = []
    for i, r in enumerate(all_results):
        walk = [a[0] for a in r["walk"]]
        walk_with_data.append({"walk": walk, "d_planning": path_length(walk, distance), "dh": path_length(all_routes_gt[i], distance), "success": True, "description": all_routes[i][1]})
    l2 = [min(1,r["dh"]/r["d_planning"]) for r in walk_with_data]
    efficiency = np.average([l2[i] for i in range(len(l2)) if walk_with_data[i]["success"]])
    print('Path Efficiency: ', efficiency)
