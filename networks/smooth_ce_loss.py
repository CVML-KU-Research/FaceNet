import torch
import torch.nn as nn
import torch.nn.functional as F


# Improved loss function with label smoothing and regularization
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        loss = (-smooth_one_hot * log_prob).sum(dim=1).mean()
        return loss
    
if __name__ == "__main__":
    # Example usage
    criterion = SmoothCrossEntropyLoss(smoothing=0.1)
    pred = torch.randn(10, 5)  # 10 samples, 5 classes
    target = torch.randint(0, 5, (10,))  # Random target labels
    loss = criterion(pred, target)
    print(f"Smooth Cross Entropy Loss: {loss.item()}")