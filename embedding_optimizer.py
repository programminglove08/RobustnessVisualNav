import torch
import torch.nn.functional as F

class EmbeddingOptimizer:
    """
    Class for optimizing image embeddings using gradient descent.

    Attributes:
        model (torch.nn.Module): Neural network model for generating embeddings.
        learning_rate (float): Learning rate for gradient descent.
    """

    def __init__(self, model, learning_rate):
        """
        Initializes the EmbeddingOptimizer with a model and learning rate.
        Args:
            model (torch.nn.Module): Model for generating embeddings.
            learning_rate (float): Learning rate for gradient descent.
        """
        self.model = model
        self.lr = learning_rate

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
        while squared_l2_distance >= l2_dist_threshold or cosine_sim <= cosine_sim_threshold:
            cur_input = cur_input.clone().detach().requires_grad_(True)

            cur_emb = self.model.encode_image(cur_input)

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

            squared_l2_distance = torch.sum((target_emb - updated_emb)**2).item()

            updated_l1_dist = torch.sum(torch.abs(updated_input.detach() - org_input)).item()
            cosine_sim = F.cosine_similarity(target_emb, updated_emb)

            print(iteration_count, '\n')
            print("Squared L2 Distance:", squared_l2_distance)
            print("Cosine Similarity:", cosine_sim.detach().cpu().item())

            l1_dist_arr.append(updated_l1_dist)
            cosine_sim_arr.append(cosine_sim.detach().cpu().item())

            cur_input = updated_input
            iteration_count += 1

        return cur_input.detach(), l1_dist_arr, cosine_sim_arr, loss_arr
