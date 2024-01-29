import argparse
import torch
from embedding_optimizer import EmbeddingOptimizer
from helper import load_image, save_image_file, inverse_normalize
import clip
import open_clip

def main():
    parser = argparse.ArgumentParser(description="Optimize Single Image Embedding")
    parser.add_argument("--current_image_path", type=str, required=True, help="File path of the current image")
    parser.add_argument("--target_image_path", type=str, required=True, help="File path of the target image")
    parser.add_argument("--learning_rate", type=float, default=0.08, help="Learning rate for gradient descent")
    parser.add_argument("--l2_dist_threshold", type=float, default=1e-4, help="Squared L2 distance threshold")
    parser.add_argument("--cosine_sim_threshold", type=float, default=0.97, help="Cosine similarity threshold")
    parser.add_argument("--output_path", type=str, required=True, help="File path to save the optimized image")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=device)
    # tokenizer = open_clip.get_tokenizer('ViT-H-14')

    optimizer = EmbeddingOptimizer(model, args.learning_rate)

    current_image = load_image(args.current_image_path, preprocess, device)
    target_image = load_image(args.target_image_path, preprocess, device)

    target_image_emb = model.encode_image(target_image)

    optimized_image, _, _, _ = optimizer.optimize_embeddings(current_image, target_image_emb, args.l2_dist_threshold, args.cosine_sim_threshold)
    optimized_image_inv = inverse_normalize()(optimized_image)

    save_image_file(optimized_image_inv, args.output_path, "optimized_image")

if __name__ == "__main__":
    main()

