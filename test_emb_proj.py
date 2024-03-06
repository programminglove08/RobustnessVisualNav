import os
import argparse
import glob
import torch
import numpy as np
import clip
from PIL import Image
import open_clip
import clip
from embedding_projection import ImageEmbeddingProjector, EmbeddingProjectionVisualizer

def main():
    parser = argparse.ArgumentParser(description="Image Embedding Processing and Visualization")
    parser.add_argument("--image_dir1", type=str, required=True, help="Directory containing original image files")
    parser.add_argument("--image_dir2", type=str, required=True, help="Directory containing image files for projection")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for visualizations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=device)
    model, preprocess = clip.load("ViT-L/14", device=device)

    # Create an instance of the ImageEmbeddingProjector
    projector = ImageEmbeddingProjector(model, preprocess, device=device)

    # Load and process images from a directory, fit PCA and get the basis vectors
    image_paths1 = glob.glob(os.path.join(args.image_dir1, '*.png'))
    image_paths1.sort()
    emb_map, emb_matrix = projector.load_and_embed_images(image_paths1)
    projector.fit_pca(emb_matrix)

    # Project the embeddings of considered images onto the principal components and store the projections
    image_paths2 = glob.glob(os.path.join(args.image_dir2, '*.png'))
    image_paths2.sort()
    emb_map, emb_matrix = projector.load_and_embed_images(image_paths2)
    emb_projection_map = projector.project_embeddings(emb_matrix, emb_map, image_paths2)
    
    img_title_map = {
        image_paths2[0]: "building",
        image_paths2[1]: "manhole cover",
        image_paths2[2]: "truck",
        image_paths2[3]: "building -> truck",
        image_paths2[4]: "manhole cover -> building",
        image_paths2[5]: "truck -> building"
    }

    texts =["building", "manhole cover", "truck"]

    mat_org = np.array([[9.995e-01, 7.963e-05, 3.352e-04],
        [2.563e-06, 1.000e+00, 0.000e+00],
        [3.755e-06, 0.000e+00, 1.000e+00]])
    
    mat_final = np.array([[2.0340e-02, 0.0000e+00, 9.7949e-01],
        [1.0000e+00, 0.0000e+00, 2.6703e-05],
        [1.0000e+00, 0.0000e+00, 8.9407e-07]])

    # Create an instance of the EmbeddingProjectionVisualizer
    visualizer = EmbeddingProjectionVisualizer(emb_projection_map, img_title_map, mat_org, mat_final, texts, args.output_path)
    # Plot the projections and matrices
    visualizer.plot_projections_and_matrices()

if __name__ == "__main__":
    main()
