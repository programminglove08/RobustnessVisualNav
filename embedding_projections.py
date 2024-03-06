import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA

class ImageEmbeddingProjector:
    def __init__(self, model, preprocess, device='cpu'):
        """
        Initializes the ImageEmbeddingProjector with a model and device.

        Args:
            model: The model used for generating image embeddings.
            device (str): The computing device ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        self.preprocess = preprocess
        self.pca = PCA()

    def inverse_normalize(self, image_tensor):
        """
        Applies inverse normalization to an image tensor.
        """
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        inv_norm = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
        return inv_norm(image_tensor)

    def load_and_embed_images(self, image_dirs):
        """
        Loads images from paths, computes their embeddings, and adds them to the embedding matrix.

        Args:
            image_dirs (list): List of paths to the images.

        Returns:
            emb_map (dict): A dictionary mapping image names to their inverse normalized image and embedding.
            emb_matrix (np.ndarray): Array of image embeddings.
        """
        emb_map = {}
        emb_matrix = []

        for i, img_path in enumerate(image_dirs):
            cur_image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cur_image_emb = self.model.encode_image(cur_image)
            
            inv_img = self.inverse_normalize(cur_image)
            emb_map[img_path] = (inv_img, cur_image_emb)
            emb_matrix.append(cur_image_emb[0].cpu().numpy())

        emb_matrix = np.asarray(emb_matrix)
        return emb_map, emb_matrix

    def fit_pca(self, emb_matrix):
        """
        Fits PCA on the embedding matrix and prints the eigenvalues and vectors.

        Args:
            emb_matrix (np.ndarray): The embedding matrix for PCA fitting.
        """
        self.pca.fit(emb_matrix)
        print('\nEigen Vectors:\n', self.pca.components_.shape)
        print('\nEigen Values:\n', self.pca.explained_variance_)
        print('\nExplained variation per principal component: {}'.format(self.pca.explained_variance_ratio_))

    def project_embeddings(self, emb_matrix, emb_map, list_imgs):
        """
        Projects embeddings onto principal components and stores the projections.

        Args:
            emb_matrix (np.ndarray): The embedding matrix to project.
            list_imgs (list): List of image names corresponding to embeddings.

        Returns:
            emb_projection_map (dict): A dictionary mapping image names to their projections.
        """
        emb_projection_map = {}
        for i, img in enumerate(list_imgs):
            emb_prj = self.pca.transform(emb_matrix[i][np.newaxis, :])
            emb_projection_map[img] = (emb_map[img][0].cpu(), emb_prj[0][0:6])
        return emb_projection_map



class EmbeddingProjectionVisualizer:
    """
    Visualizes embedding projections and corresponding vision-text matrices.
    """

    def __init__(self, emb_projection_map, img_title_map, mat_org, mat_final, texts, output_path):
        """
        Initializes the visualizer with data for plotting.

        Args:
            emb_projection_map (dict): Mapping from image identifiers to their embedding projections.
            img_title_map (dict): Mapping from image identifiers to titles for display.
            mat_org (np.array): Original vision-text matrix.
            mat_final (np.array): Vision-text matrix after transformation.
            texts (list of str): Text descriptions associated with each column of the vision-text matrices.
            output_path (str): Path to save the resulting plot.
        """
        self.emb_projection_map = emb_projection_map
        self.img_title_map = img_title_map
        self.mat_org = mat_org
        self.mat_final = mat_final
        self.texts = texts
        self.output_path = output_path

    def plot_projections_and_matrices(self):
        """
        Plots the images, their embedding projections, and vision-text matrices.
        """
        N = len(self.mat_org)
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(nrows=2, ncols=7, squeeze=False, figsize=(24, 9))
        nrows, ncols, crow, ccol = 2, 7, 0, 0

        for i, img in enumerate(self.emb_projection_map):
            # plot the image and its embedding projection
            axs[crow, ccol].set_title(self.img_title_map[img])
            axs[crow, ccol].imshow(self.emb_projection_map[img][0][0].permute(1,2,0))
            axs[crow, ccol+1].bar(range(len(self.emb_projection_map[img][1])), self.emb_projection_map[img][1], width=0.3, align='center')
            axs[crow, ccol+1].set(xticks = range(len(self.emb_projection_map[img][1])))
            axs[crow, ccol+1].set(yticks = [-6, -3, 0, 3, 6, 9])
            axs[crow, ccol+1].set_xlabel('principal component')
            axs[crow, ccol+1].set_ylabel('projected value')
            axs[crow, ccol+1].set_title('embedding projection')
            # label the subplots
            if i < len(self.img_title_map.keys())//2: axs[crow, ccol+1].text(-2.0, -13.0, '('+chr(97+i)+')')
            else: axs[crow, ccol+1].text(-2.0, -13.0, '('+chr(97+i+1)+')')
            axs[crow, ccol].axis('off')
            ccol += 2
            # set the vision-text matrices first row
            if ccol == ncols-1:
                if crow==0: matrix = self.mat_org
                elif crow: matrix = self.mat_final
                axs[crow, ccol].axis('off')
                axs[crow, ccol].set_title('vision x text', x = 0.5, y = 0.5)
                if crow==0: 
                    axs[crow, ccol].text(0.85, -1.9, '('+chr(97+i+1)+')')
                    axs[crow, ccol].text(-0.9, 3.4, 'Image')
                    axs[crow, ccol].text(-0.65, 2.4, '('+chr(97+i-2)+')')
                    axs[crow, ccol].text(-0.65, 1.4, '('+chr(97+i-1)+')')
                    axs[crow, ccol].text(-0.65, 0.4, '('+chr(97+i-0)+')')
                    axs[crow, ccol].text(0.10, 3.4, self.texts[i-2], fontsize=9)
                    axs[crow, ccol].text(1.04, 3.4, self.texts[i-1], fontsize=5.5)
                    axs[crow, ccol].text(2.10, 3.4, self.texts[i-0], fontsize=9)
                elif crow: 
                    axs[crow, ccol].text(0.85, -1.9, '('+chr(97+i+2)+')')
                    axs[crow, ccol].text(-0.9, 3.4, 'Image')
                    axs[crow, ccol].text(-0.65, 2.4, '('+chr(97+i-1)+')')
                    axs[crow, ccol].text(-0.65, 1.4, '('+chr(97+i-0)+')')
                    axs[crow, ccol].text(-0.65, 0.4, '('+chr(97+i+1)+')')
                    axs[crow, ccol].text(0.10, 3.4, self.texts[i-5], fontsize=9)
                    axs[crow, ccol].text(1.04, 3.4, self.texts[i-4], fontsize=5.5)
                    axs[crow, ccol].text(2.10, 3.4, self.texts[i-3], fontsize=9)
                axs[crow, ccol].matshow(matrix, cmap='gray_r', vmin= 0, vmax = 0, extent=[-1, N, 0, N+1])

                # draw the grid
                for x in range(N + 1):
                        axs[crow, ccol].axhline(x, lw=2, color='k')
                        axs[crow, ccol].axvline(x, lw=2, color='k')
                axs[crow, ccol].axhline(4, lw=2, color='k')
                axs[crow, ccol].axvline(-1, lw=2, color='k')

                # annotate the matrix values
                for i in range(len(matrix)):
                        for j in range(len(matrix)):
                                val = matrix[j, i]
                                if val < 0.5: axs[crow, ccol].text(i+0.12, -j+N-1+0.4, str(val), fontsize=7)
                                else: axs[crow, ccol].text(i+0.12, -j+N-1+0.4, str(val), fontsize=7)

                ccol =0
                crow += 1

        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.show()