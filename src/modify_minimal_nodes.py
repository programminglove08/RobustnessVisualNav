# The core code was taken from https://github.com/blazejosinski/lm_nav repository and then modified to our need

from lm_nav.navigation_graph import NavigationGraph
from lm_nav.utils import rectify_and_crop
from pathlib import Path
import numpy as np
import io
from PIL import Image
import random
from tqdm import tqdm
import clip
from torchvision.transforms.functional import to_pil_image
import PIL
import copy
from lm_nav.optimal_route import nodes_landmarks_similarity
from collections import deque
from lm_nav.optimal_route import dijskra_transform
import heapq
import torch
from embedding_optimizer import EmbeddingOptimizer
from helper import preprocess_image
from torchvision import transforms
import toml
import copy
import heapq

def dijkstra_shortest_path(graph, start: int, target: int):
    distances = {node: float('inf') for node in range(graph.vert_count)}
    distances[start] = 0
    
    predecessors = {node: None for node in range(graph.vert_count)}
    
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_node == target:
            break  # Stop once we reach the target
        
        for neighbor in graph._graph.neighbors(current_node):
            weight = graph._graph.get_edge_data(current_node, neighbor)["weight"]
            distance_through_current = current_distance + weight
            
            if distance_through_current < distances[neighbor]:
                distances[neighbor] = distance_through_current
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance_through_current, neighbor))
                
    return distances, predecessors

def reconstruct_path(predecessors, start: int, target: int):
    path = []
    current_node = target
    while current_node is not None and current_node != start:
        path.append(current_node)
        current_node = predecessors[current_node]
    if current_node is None:
        return []  # No path found
    path.append(start)
    path.reverse()
    return path

def find_nodes_on_shortest_path(graph, start: int, target: int):
    distances, predecessors = dijkstra_shortest_path(graph, start, target)
    path = reconstruct_path(predecessors, start, target)
    return path

def find_best_landmark_nodes_in_shortest_path(similarities, path):

    filtered_result = similarities[path, :]


    max_values_filtered = np.max(filtered_result, axis=0)


    max_indices_filtered = np.argmax(filtered_result, axis=0)


    max_indices_original = [path[i] for i in max_indices_filtered]
    thresholds = []
    landmarks_in_path = []

    for i, (value, index) in enumerate(zip(max_values_filtered, max_indices_original)):
        thresholds.append(value)
        landmarks_in_path.append(index)
    return thresholds, landmarks_in_path

def max_similarity_with_path(sim_matrix, shortest_path):
    similarities = copy.deepcopy(sim_matrix)
    similarities = similarities[shortest_path]
    m, n = len(similarities), len(similarities[0])
    if n > m:
        result = [i for i in range(m)]
        for j in range(m, n):
            result.append(m-1)
        max_sum = 0
        for i, row in enumerate(result):
            max_sum += similarities[row][i]
        thresholds = [similarities[row_indice][i] for i, row_indice in enumerate(result)]
        landmarks_in_path = [shortest_path[i] for i in result]
        return thresholds, landmarks_in_path
        
    max_matrix = [[-float('inf')] * n for _ in range(m)]

    path = [[-1] * n for _ in range(m)]
    

    for i in range(m):
        max_matrix[i][0] = similarities[i][0]
    

    for j in range(1, n):
        for i in range(m):
            for k in range(i):
                if max_matrix[i][j] < max_matrix[k][j-1] + similarities[i][j]:
                    max_matrix[i][j] = max_matrix[k][j-1] + similarities[i][j]
                    path[i][j] = k  
    

    max_sum = -float('inf')
    last_row = -1
    for i in range(m):
        if max_matrix[i][n-1] > max_sum:
            max_sum = max_matrix[i][n-1]
            last_row = i
    

    row_indices = [-1] * n
    current_row = last_row
    for j in range(n-1, -1, -1):
        row_indices[j] = current_row
        current_row = path[current_row][j]

    thresholds = [similarities[row_indice][i] for i, row_indice in enumerate(row_indices)]
    landmarks_in_path = [shortest_path[i] for i in row_indices]
    return thresholds, landmarks_in_path

def find_landmark_path(sim_matrix, shortest_path):
    similarity = copy.deepcopy(sim_matrix)
    similarity = similarity[shortest_path]
    m, n = len(similarity), len(similarity[0])

    if n > m:
        result = [i for i in range(m)]
        for j in range(m, n):
            result.append(m-1)
        max_sum = 0
        for i, row in enumerate(result):
            max_sum += similarity[row][i]
        thresholds = [similarity[row_indice][i] for i, row_indice in enumerate(result)]
        landmarks_in_path = [shortest_path[i] for i in result]
        return thresholds, landmarks_in_path

    dp = [[float('-inf') for _ in range(n+1)] for _ in range(m+1)]
    dp[0][0] = 0
    

    parent = [[None for _ in range(n+1)] for _ in range(m+1)]
    

    for i in range(1, m+1):
        for j in range(n+1):

            if dp[i-1][j] > dp[i][j]:
                dp[i][j] = dp[i-1][j]
                parent[i][j] = (i-1, j)
                

            if j > 0 and dp[i-1][j-1] + similarity[i-1][j-1] > dp[i][j]:
                dp[i][j] = dp[i-1][j-1] + similarity[i-1][j-1]
                parent[i][j] = (i-1, j-1)
    
 
    selected_nodes = []
    thresholds = []
    i, j = m, n
    while j > 0:
        prev_i, prev_j = parent[i][j]
        if prev_j < j:  
            selected_nodes.append(prev_i)
            thresholds.append(similarity[prev_i][prev_j])
        i, j = prev_i, prev_j
    

    selected_nodes.reverse()
    thresholds.reverse()
    landmarks_in_path = [shortest_path[i] for i in selected_nodes]
    return  thresholds, landmarks_in_path

def inverse_normalize():
    """
    Returns an inverse normalization transformation.
    Returns:
        transforms.Normalize: Transformation to apply inverse normalization.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

def optimize_image_multiple(current_image_data, target_image_data_list, l2_dist_threshold, cosine_sim_threshold, model, preprocess, optimizer, device):
    current_image = preprocess_image(current_image_data, preprocess, device)
    target_embs = []
    for target_image_data in target_image_data_list:
        target_image = preprocess_image(target_image_data, preprocess, device)

        target_image_emb = model.encode_image(target_image)
        target_embs.append(target_image_emb)
    optimized_image, _, _, _ = optimizer.optimize_embeddings_multiple(current_image, target_embs, l2_dist_threshold, cosine_sim_threshold)
    optimized_image_inv = inverse_normalize()(optimized_image)
    return optimized_image_inv

def optimize_image_with_text(current_image_data, target_text_embeddings, l2_dist_threshold, cosine_sim_threshold, model, preprocess, optimizer, device):
    current_image = preprocess_image(current_image_data, preprocess, device)
    target_text_embeddings = [target_text_embedding.unsqueeze(0) for target_text_embedding in target_text_embeddings]
    optimized_image, _, _, _ = optimizer.optimize_embeddings_multiple(current_image, target_text_embeddings, l2_dist_threshold, cosine_sim_threshold)
    optimized_image_inv = inverse_normalize()(optimized_image)
    return optimized_image_inv

def img_to_img_attack(copy_graph, input_node, target_nodes, img_to_img_args, model, preprocess, device):
    new_images = copy.deepcopy(copy_graph._images)
    args = img_to_img_args
    optimizer = EmbeddingOptimizer(model, args["learning_rate"], device)
    input_images = copy_graph._images[input_node]
    target_images = [copy_graph._images[i] for i in target_nodes]
    for j, img_bytes in enumerate(input_images):
        current_image_data = copy.deepcopy(input_images[j])
        target_image_data_list = copy.deepcopy([img[j] for img in target_images])

        already_recitified_current_image = PIL.Image.fromarray(np.array(PIL.Image.open(io.BytesIO(current_image_data))))
        already_recitified_target_images = [
            PIL.Image.fromarray(np.array(PIL.Image.open(io.BytesIO(target_image_data)))) for target_image_data in target_image_data_list]
        
        preprocessed_image_tensor = optimize_image_multiple(already_recitified_current_image, already_recitified_target_images,
                                                    args["l2_dist_threshold"],
                                                    args["cosine_sim_threshold"],
                                                    model,
                                                    preprocess,
                                                    optimizer,
                                                    device)
        img_reversed = to_pil_image(preprocessed_image_tensor.squeeze(0).cpu())
        
        with io.BytesIO() as output:
            img_reversed.save(output, format='PNG')
            new_images[input_node][j] = output.getvalue()

        copy_graph._images = new_images
    return copy_graph

def img_to_text_attack(copy_graph, input_node, target_landmarks, img_to_txt_args,  model, preprocess, device):

    args = img_to_txt_args

    tokenizer = clip.tokenize  
    optimizer = EmbeddingOptimizer(model, args["learning_rate"], device)

    target_texts = ["A photo of " + desc for desc in target_landmarks]
    text_embeddings = optimizer.get_text_embeddings(target_texts, tokenizer)
    new_images = copy.deepcopy(copy_graph._images)
    input_images = copy_graph._images[input_node]
    for j, img_bytes in enumerate(input_images):
        current_image_data = copy.deepcopy(input_images[j])

        already_recitified_current_image = Image.fromarray(np.array(Image.open(io.BytesIO(current_image_data))))

        preprocessed_image_tensor = optimize_image_with_text(
            already_recitified_current_image,
            text_embeddings,  
            args["l2_dist_threshold"],
            args["cosine_sim_threshold"],
            model,
            preprocess,
            optimizer,
            device
        )

        img_reversed = to_pil_image(preprocessed_image_tensor.squeeze(0).cpu())
        

        with io.BytesIO() as output:
            img_reversed.save(output, format='PNG')
            new_images[input_node][j] = output.getvalue()

        copy_graph._images = new_images
    return copy_graph

def find_nodes_to_decrease(similarities, thresholds, landmarks_in_path, threshold_const=0.97):
    indexes_per_column = []

    for i in range(len(landmarks_in_path)):
        column_values = similarities[:, i]  # Get the i-th column
        threshold = thresholds[i]  # Get the i-th threshold
        indexes = np.where(column_values >= threshold*threshold_const)[0]  # Find indexes meeting the threshold in this column
        indexes = list(indexes)

        for l in landmarks_in_path:
            if l in indexes:
                indexes.remove(l)
        indexes_per_column.append(indexes)


    for i, indexes in enumerate(indexes_per_column):
        print(f"Indexes in column {i} where value is >= {thresholds[i]}:", indexes)
    return indexes_per_column

def optimize_minimal_nodes(graph, start_node, landmarks, target_node, model, preprocess, img_to_img_args, img_to_txt_args, device):

    copy_graph = copy.deepcopy(graph)

    path = find_nodes_on_shortest_path(copy_graph, start_node, target_node)
    print(f'From starting node {start_node} to target node {target_node} shortest path contains these nodes: {path}')

    similarities = nodes_landmarks_similarity(copy_graph, landmarks)
    thresholds, landmarks_in_path = find_landmark_path(similarities, path)
    print(f'Best nodes for each landmark in the path: {landmarks_in_path}. Similarities: {thresholds}')

    thresholds[-1] = similarities[target_node][-1]
    landmarks_in_path[-1] = target_node
    print(f'Replacing last landmark node with target node...\nBest nodes for each landmark in the path: {landmarks_in_path}. Similarities: {thresholds}')

    
    best_matches = np.argmax(similarities, axis=0)
    worst_matches = np.argmin(similarities, axis=0)

    print(f'Worst matches: ', worst_matches)
    increased_nodes = []
    decreased_nodes = []
    increased_targets = []
    decreased_targets = []
    decreased_groups = []
    # Optimize the chosen ladnmark nodes again with the landmark text based embedding. For landmark i optimize them with 0 to ith landmark texts
    for i, l in enumerate(landmarks_in_path):
        
        increased_nodes.append(l)
        increased_targets.append(landmarks[:i+1])
        if i == len(landmarks_in_path):
            print(f'Optimizing node {l} with {landmarks[:i+1]} for increasing similarity...')
            copy_graph = img_to_text_attack(copy_graph, l, landmarks[:i+1], img_to_txt_args, model, preprocess, device)
        else:
            print(f'Optimizing node {l} with {[landmarks[i]]} for increasing similarity...')
            copy_graph = img_to_text_attack(copy_graph, l, [landmarks[i]], img_to_txt_args, model, preprocess, device)

    modified_similarities = nodes_landmarks_similarity(copy_graph, landmarks)
    modified_thresholds = [modified_similarities[node, i] for i, node in enumerate(landmarks_in_path)]

    nodes_to_decrease = find_nodes_to_decrease(modified_similarities, modified_thresholds, landmarks_in_path)
    
    test_sims = nodes_landmarks_similarity(copy_graph, landmarks)

    print(f'Before optimization: Nodes: {landmarks_in_path} Similarities: {similarities[landmarks_in_path]}')
    print(f'After optimization: Nodes: {landmarks_in_path} Similarities: {test_sims[landmarks_in_path]}')
    
    for i in range(len(nodes_to_decrease)):
        decreased_groups.append(nodes_to_decrease[i])
        for j in nodes_to_decrease[i]:
            print(f'Optimizing node {j} with {[worst_matches[i]]} for decreasing similarity...')
            decreased_nodes.append(j)
            decreased_targets.append(worst_matches[i])
            copy_graph = img_to_img_attack(copy_graph, j, [worst_matches[i]], img_to_img_args, model, preprocess, device)
    
    data = {"start_node": start_node,
            "target_node": target_node,
            "landmarks": landmarks, 
            "increased_nodes": increased_nodes, 
            "increased_targets": increased_targets,
            "decreased_nodes": decreased_nodes,
            "decreased_targets": decreased_targets, 
            "decreased_groups": decreased_groups}
    return copy_graph, data
    
def main(inx, original_graph, landmarks, start_node, target_node, img_to_img_args, img_to_txt_args,device, model, preprocess):
    modified_graph, data = optimize_minimal_nodes(original_graph, start_node, landmarks, target_node, model, preprocess, img_to_img_args, img_to_txt_args, device)
    modified_graph.save_to_file(f'experiments/large_graphs/large_inx{inx}_to_node80.pkl')
    file_path = f"experiments/large_graph_infos/info_large_inx{inx}_to_node80.toml"

    with open(file_path, "w") as toml_file:
        toml.dump(data, toml_file)
if __name__ == "__main__":
    # Large Graph # Similar for Small Graph
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
    all_routes_gt = [
        [5, 8, 77],
        [5, 8, 23, 261, 77],
        [173, 160, 150, 191, 129, 45],
        [108, 210, 217, 220],
        [247, 254, 264, 275],
        [70, 39, 34, 257],
        [215, 194, 184, 170],
        [103, 267, 22, 8],
        [103, 16, 8],
        [211, 220, 217, 204],
    ]
    landmarks_cache = [['a white building', 'a white truck', 'a stop sign'], ['a white building', 'a white truck', 'a square with a large tree', 'a stop sign'], ['a building with a red-black wall', 'a fire hydrant', 'a grove', 'a manhole cover', 'a trailer'], ['a stop sign', 'a white car', 'a glass building'], ['a blue semi-truck', 'an orange traffic cone', 'a blue dumpster', 'a picnic bench'], ['a white trailer', 'traffic lights', 'a traffic cone'], ['a stop sign', 'a manhole cover', 'a disabled Parking spot', 'a red building'], ['a stop sign', 'a square with a tree', 'a white truck', 'a white building'], ['a stop sign', 'a white truck', 'a white building'], ['a glass building', 'a white car', 'a stop sign']]

    original_graph = NavigationGraph("graphs/graph_rectified_cropped.pkl")

    img_to_img_args = {"learning_rate": 0.9, "l2_dist_threshold": 25, "cosine_sim_threshold": .95}
    img_to_txt_args = {"learning_rate": 0.9, "l2_dist_threshold": 269, "cosine_sim_threshold": .75}
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    for i in tqdm(range(0, len(all_routes))):
        start_node = all_routes[i][0]
        landmarks = landmarks_cache[i]
        target_node = 80
        main(i, copy.deepcopy(original_graph), landmarks, start_node, target_node, img_to_img_args, img_to_txt_args, device, model, preprocess)


