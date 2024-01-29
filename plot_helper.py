import matplotlib.pyplot as plt

class ImagePlotter:
    def __init__(self, figure_size=(24, 5)):
        self.figure_size = figure_size

    def plot_matched_images(self, image_tensors, image_title_map, output_path):
        plt.rcParams.update({'font.size': 10})
        fig, axs = plt.subplots(nrows=1, ncols=len(image_tensors), squeeze=False, figsize=self.figure_size)
        for i, img_tensor in enumerate(image_tensors):
            title = image_title_map.get(i, '')
            axs[0, i].set_title(title)
            axs[0, i].imshow(img_tensor.cpu().permute(1, 2, 0))
            axs[0, i].axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

    def plot_images_and_matrix(self, image_tensor_list, matrix, img_title_map, output_path):
        N = len(matrix)
        fig, axs = plt.subplots(nrows=1, ncols=N+1, figsize=self.figure_size)

        for i, img_tensor in enumerate(image_tensor_list):
            axs[i].set_title(img_title_map.get(i, ''))
            axs[i].imshow(img_tensor.cpu().permute(1, 2, 0))
            axs[i].axis('off')

        self._plot_matrix(axs[-1], matrix)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

    def _plot_matrix(self, ax, matrix):
        N = len(matrix)
        ax.axis('off')
        ax.set_title('Vision x Text', x=0.5, y=0.5)
        ax.matshow(matrix, cmap='gray_r', vmin=0, vmax=1, extent=[0, N+1, 0, N])

        for x in range(N + 1):
            ax.axhline(x, lw=2, color='k')
        for x in range(N + 2):
            ax.axvline(x, lw=2, color='k')

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                val = matrix[i, j]
                text_color = 'black' if val < 0.5 else 'white'
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=7)
                
                
                
class MatrixPlotter:
    def __init__(self, matrix, output_path):
        self.matrix = matrix
        self.output_path = output_path

    def plot_vision_text_matrix(self):
        fig, ax = plt.subplots(figsize=(15, 11))
        N = len(self.matrix)
        ax.matshow(self.matrix, cmap='gray_r', vmin=0, vmax=1, extent=[0, N+1, 0, N])

        for x in range(N + 1):
            ax.axhline(x, lw=2, color='k')
        for x in range(N + 2):
            ax.axvline(x, lw=2, color='k')

        for i in range(N):
            for j in range(N):
                val = self.matrix[i, j]
                text_color = 'black' if val < 0.5 else 'white'
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.show()

