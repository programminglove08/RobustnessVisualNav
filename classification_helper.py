import torch
from models import data
from models import imagebind_model
from models.imagebind_model import ModalityType
from PIL import Image

class ClassificationHelper:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)

    def get_vision_text_matrix(self, image_paths, text_list):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, self.device),
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        vision_text_matrix = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
        # print("Vision x Text: \n", vision_text_matrix.cpu())
        return vision_text_matrix.cpu()
        
        
class VisionTextMatrixGenerator:
    def __init__(self, model, processor, tokenizer, device):
        """
        Initializes the VisionTextMatrixGenerator with a model, processor, tokenizer, and device.

        Args:
            model: Model to generate image and text features.
            processor: Processor for image data.
            tokenizer: Tokenizer for text data.
            device: Device on which to perform computations ('cuda' or 'cpu').
        """
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device

    def generate_matrix(self, image_paths, texts):
        """
        Generates a vision-text similarity matrix for given images and texts.

        Args:
            image_paths (list of str): Paths to the image files.
            texts (list of str): Texts to compare with the images.

        Returns:
            torch.Tensor: The vision-text similarity matrix.
        """
        images = [Image.open(path) for path in image_paths]
        processed_images = self.processor(images=images, return_tensors="pt").to(self.device)
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)

        image_features = self.model.get_image_features(processed_images['pixel_values'])
        text_features = self.model.get_text_features(tokenized_texts['input_ids'])

        vision_text_matrix = torch.softmax(image_features @ text_features.T, dim=-1)
        return vision_text_matrix.cpu()


