import argparse
import torch
import clip
from PIL import Image
import glob
import open_clip
from embedding_optimizer import EmbeddingOptimizer
from helper import save_images, inverse_normalize

def main():
    parser = argparse.ArgumentParser(description="Optimize Image Embeddings")
    parser.add_argument("--learning_rate", type=float, default=5, help="Learning rate for gradient descent")
    parser.add_argument("--input_dir", type=str, default="./images_directory", help="Directory of input images")
    parser.add_argument("--output_dir", type=str, default="./output_directory", help="Directory to save optimized images")
    parser.add_argument("--l2_dist_threshold", type=float, default=1e-4, help="Squared L2 distance threshold")
    parser.add_argument("--cosine_sim_threshold", type=float, default=0.99, help="Cosine similarity threshold")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=device)
    # tokenizer = open_clip.get_tokenizer('ViT-H-14')

    optimizer = EmbeddingOptimizer(model, args.learning_rate)

    dirs = glob.glob(f"{args.input_dir}/*.jpeg")
    dirs.sort()
    target_img = preprocess(Image.open(dirs[0])).unsqueeze(0).to(device)
    target_img_emb = model.encode_image(target_img)

    optimized_images = []
    for img_path in dirs[1:]:
        cur_img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        optimized_img, _, _, _ = optimizer.optimize_embeddings(cur_img, target_img_emb, args.l2_dist_threshold, args.cosine_sim_threshold)
        optimized_img = inverse_normalize()(optimized_img)
        optimized_images.append(optimized_img)

    save_images(optimized_images, args.output_dir, "optimized_image")

if __name__ == "__main__":
    main()

