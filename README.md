# Robustness of Vision-Language Models for Visual Navigation.

This is the official repository for the experiments for our work on exploring the Robustness of Vision-Language Models for Visual Navigation. The code will be made publicly available once cleaned up.
Note that the optimization method was introduced first in our earlier work titled as "Intriguing Equivalence Structures of the Embedding Space of Vision Transformers". This paper is in arXiv, and the GitHub link for the code: https://github.com/programminglove08/EquivalenceStruct. 

![Screenshot](example_plot.png)

Example run for the codes: 

python test1.py --current_image_path "./path/to/current_image.jpeg"
--target_image_path "./path/to/target_image.jpeg"
--learning_rate 0.09
--l2_dist_threshold 16
--cosine_sim_threshold 0.96
--output_path "./path/to/output_image"

python test2.py --learning_rate 0.09 --input_dir "./path/to/input_images" --output_dir "./path/to/output_images" --l2_dist_threshold 16 --cosine_sim_threshold 0.96

python test_emb_proj.py --image_dir1 "./images1" --image_dir2 "./images2" --output_path "./path/to/output/projections.pdf"

# Robot Route Manipulation Code

This codebase is designed for manipulating robot routes based on VLM (Visual Language Model) with the aid of representational similarity matching. It employs two types of representations (embeddings) for route manipulation: Target Image Embedding and Target Text Embedding.

The LM-Nav System's source code was originally obtained from the following repository: LM-Nav GitHub Repository (https://github.com/blazejosinski/lm_nav/tree/main). Modifications were made to adapt the code to our specific requirements.

## Installation 

1. Clone the repository: `git clone https://github.com/programminglove08/RobustnessVisualNav.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage

This section provides detailed instructions on loading the graph, preparing the graph for manipulation, running the manipulation algorithm, and detecting manipulated routes.

### Loading graph: 

To download and load the graph, use the following instructions along with the provided code segment:

```
# Obtained from from LM-Nav
# Large graph: graph_file_gdrive_id = "186_WuE3caY0ADoaJrGurs9PACLsgSjup"
# Samll graph: graph_file_gdrive_id = "1SJiZmFDLcCnBGz2XuOLPewR1hz8ScwHi"

url = f'https://drive.google.com/uc?id={graph_file_gdrive_id}'
gdown.download(url, "graph.pkl")
graph = NavigationGraph("graph.pkl")
```

You may store the graph in the graphs folder for subsequent preparation. To save the graph, utilize the save_to_file() function in NavigationGraph.

### Prepare Graph For Manipulation: 

Before commencing manipulation, it's necessary to rectify and crop all the graph images. Utilize the following code in src/graph_modifications.ipynb:

```
graph_recitified_cropped = recity_crop_modification(graph)
```

### Path Manipulation: 
To modify different paths, refer to the src/modify_minimal_modes.py file. Example codes for loading and modifying paths are provided at the end of the file. Adjust these according to your requirements.

### Detection method: 

For detecting manipulation results, you will need two graphs: graph_rectified_and_cropped and manipulated_graph. To observe the effects of noise and magnitude representation differences, employ the src/detection.ipynb file.

By following these instructions, you can effectively manipulate and analyze robot routes using our developed codebase.