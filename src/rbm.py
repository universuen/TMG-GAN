import torch

from src import config, datasets


class RBM:

    def __init__(
            self,
            v_num: int,
            h_num: int,
    ):
        self.v_num = v_num
        self.h_num = h_num

        self.weights = torch.randn(v_num, h_num) * 0.1
        self.visible_bias = torch.ones(v_num) * 0.5
        self.hidden_bias = torch.zeros(h_num)

        self.weights_momentum = torch.zeros(v_num, h_num)
        self.visible_bias_momentum = torch.zeros(v_num)
        self.hidden_bias_momentum = torch.zeros(h_num)

        self.weights = self.weights.to(config.device)
        self.visible_bias = self.visible_bias.to(config.device)
        self.hidden_bias = self.hidden_bias.to(config.device)
        self.weights_momentum = self.weights_momentum.to(config.device)
        self.visible_bias_momentum = self.visible_bias_momentum.to(config.device)
        self.hidden_bias_momentum = self.hidden_bias_momentum.to(config.device)

    def sample_hidden(self, visible_probabilities: torch.Tensor):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = torch.sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities: torch.Tensor):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = torch.sigmoid(visible_activations)
        return visible_probabilities

    def fit(
            self,
            x: torch.Tensor,
    ):
        for _ in range(config.rbm_config.epochs):
            # Positive phase
            positive_hidden_probabilities = self.sample_hidden(x)
            positive_hidden_activations = (
                    positive_hidden_probabilities >= torch.randn(self.h_num, device=config.device)
            ).float()
            positive_associations = torch.matmul(x.t(), positive_hidden_activations)

            # Negative phase
            hidden_activations = positive_hidden_activations

            visible_probabilities = None
            hidden_probabilities = None
            for step in range(2):
                visible_probabilities = self.sample_visible(hidden_activations)
                hidden_probabilities = self.sample_hidden(visible_probabilities)
                hidden_activations = (hidden_probabilities >= torch.rand(self.h_num, device=config.device)).float()

            negative_visible_probabilities = visible_probabilities
            negative_hidden_probabilities = hidden_probabilities

            negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

            # Update parameters
            self.weights_momentum *= config.rbm_config.momentum_coefficient
            self.weights_momentum += (positive_associations - negative_associations)

            self.visible_bias_momentum *= config.rbm_config.momentum_coefficient
            self.visible_bias_momentum += torch.sum(x - negative_visible_probabilities, dim=0)

            self.hidden_bias_momentum *= config.rbm_config.momentum_coefficient
            self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

            batch_size = x.size(0)

            self.weights += self.weights_momentum * config.rbm_config.lr / batch_size
            self.visible_bias += self.visible_bias_momentum * config.rbm_config.lr / batch_size
            self.hidden_bias += self.hidden_bias_momentum * config.rbm_config.lr / batch_size

            self.weights -= self.weights * config.rbm_config.weight_decay
