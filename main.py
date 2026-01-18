import torch
import torch.nn as nn
from train import train_step

# Dummy modules

class DummyVision(nn.Module):
    def forward(self, x):
        return torch.randn(x.size(0), 512)

class DummyText(nn.Module):
    def forward(self, texts):
        return torch.randn(len(texts), 384)

class DummyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512 + 384, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, img_vec, txt_vec):
        x = torch.cat([img_vec, txt_vec], dim=1)
        return self.net(x)

# Run test

if __name__ == "__main__":
    BATCH_SIZE = 2

    vision = DummyVision()
    text_model = DummyText()
    predictor = DummyPredictor()

    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    image  = torch.randn(BATCH_SIZE, 3, 224, 224)
    text   = ["cat", "dog"]
    target = torch.randn(BATCH_SIZE, 256)

    loss = train_step(
        image,
        text,
        target,
        vision,
        text_model,
        predictor,
        optimizer
    )

    print(f"Phase 1 Test Passed | Loss: {loss:.4f}")
