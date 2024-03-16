import torch
import torch.nn.functional as F
from tqdm import tqdm

class EmbeddingOptimizer:
    """
    Class for optimizing image embeddings using gradient descent.

    Attributes:
        model (torch.nn.Module): Neural network model for generating embeddings.
        learning_rate (float): Learning rate for gradient descent.
    """

    def __init__(self, model, learning_rate, device):
        """
        Initializes the EmbeddingOptimizer with a model and learning rate.
        Args:
            model (torch.nn.Module): Model for generating embeddings.
            learning_rate (float): Learning rate for gradient descent.
        """
        self.model = model
        self.lr = learning_rate
        self.device = device

    def get_text_embeddings(self, list_texts, tokenizer):
        tokenized_texts = tokenizer(list_texts).to(self.device)
        with torch.no_grad():
            text_embs = self.model.encode_text(tokenized_texts)
        #text_embs /= text_embs.norm(dim=-1, keepdim=True)
        for i, text_emb in enumerate(text_embs):
            text_embs[i] = text_emb.reshape(1, -1)
        return text_embs
    
    def scale_tensor(self, input_tensor, target_tensor):
        min1, max1 = input_tensor.min(), input_tensor.max()
        min2, max2 = target_tensor.min(), target_tensor.max()

        # Scale tensor1 to the range of tensor2
        input_tensor_scaled = min2 + ((input_tensor - min1) * (max2 - min2)) / (max1 - min1)
        return input_tensor_scaled

    def optimize_embeddings(self, cur_input, target_emb, l2_dist_threshold, cosine_sim_threshold):
        """
        Adjusts initial input to match target embedding, stopping when thresholds are met.
        Args:
            cur_input (torch.Tensor): Input tensor to be optimized.
            target_emb (torch.Tensor): Target embedding to match.
            l2_dist_threshold (float): Threshold for squared L2 distance.
            cosine_sim_threshold (float): Threshold for cosine similarity.
        Returns:
            torch.Tensor: Optimized input tensor.
            list: L1 distances over iterations.
            list: Cosine similarities over iterations.
            list: Losses over iterations.
        """
        org_input = cur_input.clone()
        max_val = max(cur_input.flatten()).item()
        min_val = min(cur_input.flatten()).item()

        squared_l2_distance = float('inf')
        cosine_sim_arr = []
        loss_arr = []
        l1_dist_arr = []

        iteration_count = 0
        pbar = tqdm(range(15000), desc='Initializing')
        for i in pbar:
            if not (squared_l2_distance >= l2_dist_threshold or cosine_sim <= cosine_sim_threshold):
                break
        #while squared_l2_distance >= l2_dist_threshold or cosine_sim <= cosine_sim_threshold:
            cur_input = cur_input.clone().detach().requires_grad_(True)

            cur_emb = self.model.encode_image(cur_input)
            #cur_emb /= cur_emb.norm(dim=-1, keepdim=True)

            loss = F.mse_loss(target_emb, cur_emb)
            loss_arr.append(loss.item())

            cur_input.grad = None
            loss.backward(retain_graph=True)
            grad = cur_input.grad

            updated_input = cur_input - self.lr * grad
            updated_input[updated_input > max_val] = max_val
            updated_input[updated_input < min_val] = min_val

            with torch.no_grad():
                updated_emb = self.model.encode_image(updated_input)
                #updated_emb /= updated_emb.norm(dim=-1, keepdim=True) 
            squared_l2_distance = torch.sum((target_emb - updated_emb)**2).item()

            updated_l1_dist = torch.sum(torch.abs(updated_input.detach() - org_input)).item()
            cosine_sim = F.cosine_similarity(target_emb, updated_emb)
            pbar.set_description(f"Iter: {i}, L2 Dist: {squared_l2_distance:.4f}, Cos Sim: {cosine_sim.detach().cpu().item():.4f}")


            l1_dist_arr.append(updated_l1_dist)
            cosine_sim_arr.append(cosine_sim.detach().cpu().item())

            cur_input = updated_input
            iteration_count += 1

        return cur_input.detach(), l1_dist_arr, cosine_sim_arr, loss_arr

    def optimize_embeddings_multiple(self, cur_input, target_embs, l2_dist_threshold, cosine_sim_threshold, loss_const=0.3):
        """
        Adjusts initial input to match multiple target embeddings, stopping when thresholds are met.
        Args:
            cur_input (torch.Tensor): Input tensor to be optimized.
            target_embs (torch.Tensor): Target embeddings to match.
            l2_dist_threshold (float): Threshold for squared L2 distance.
            cosine_sim_threshold (float): Threshold for cosine similarity.
        Returns:
            torch.Tensor: Optimized input tensor.
            list: L1 distances over iterations.
            list: Cosine similarities over iterations (average over targets).
            list: Losses over iterations.
        """
        org_input = cur_input.clone()
        max_val = max(cur_input.flatten()).item()
        min_val = min(cur_input.flatten()).item()

        loss_arr = []
        l1_dist_arr = []
        cosine_sim_arr = []

        iteration_count = 0

        cur_input = cur_input.clone().detach().requires_grad_(True)
        cur_emb = self.model.encode_image(cur_input)
        target_embs = [self.scale_tensor(target_emb, cur_emb) for target_emb in target_embs]

        pbar = tqdm(range(12000), desc='Initializing')
        for i in pbar:
            cur_input = cur_input.clone().detach().requires_grad_(True)

            cur_emb = self.model.encode_image(cur_input)
            
            n = len(target_embs) 

            # Compute each loss and scale it: 1/n for the first loss, 2/n for the second, ..., n/n for the nth loss
            scaled_losses = [(idx + 1) / n * F.mse_loss(cur_emb, target_emb) for idx, target_emb in enumerate(target_embs)]

            loss = sum(scaled_losses) / len(scaled_losses)
            loss_arr.append(loss.item())


            cur_input.grad = None
            loss.backward(retain_graph=True)
            grad = cur_input.grad

            updated_input = cur_input - self.lr * grad
            updated_input = torch.clamp(updated_input, min_val, max_val)

            with torch.no_grad():
                updated_emb = self.model.encode_image(updated_input)

            last_target_emb = target_embs[-1]
            squared_l2_distance_last = torch.sum((last_target_emb - updated_emb) ** 2).item()
            cosine_sim_last = F.cosine_similarity(last_target_emb, updated_emb).item()

            updated_l1_dist = torch.sum(torch.abs(updated_input.detach() - org_input)).item()

            pbar.set_description(f"Iter: {i}, L2 Dist (last): {squared_l2_distance_last:.4f}, Cos Sim (last): {cosine_sim_last:.4f}")

            l1_dist_arr.append(updated_l1_dist)
            cosine_sim_arr.append(cosine_sim_last)

            cur_input = updated_input
            iteration_count += 1
            if squared_l2_distance_last < l2_dist_threshold and cosine_sim_last > cosine_sim_threshold:
                break

        return cur_input.detach(), l1_dist_arr, cosine_sim_arr, loss_arr
