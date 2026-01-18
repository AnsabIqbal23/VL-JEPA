import torch
import torch.nn as nn
import timm

class VisionModule(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', output_dim=512):
        super().__init__()

        print(f"Loading pre-trained model: {model_name}")

        # num_classes=0 to remove the neural networks head, meaning it's "answer"
        # for what is the "correct class", that is not needed here.
        # Only the feature extractor part is needed. To get the vector representation.
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )

        # Freeze all backbone weights (important constraint)
        # This makes it so we do not update the weights of the pre-trained model
        # Meaning the neural network wont do back propogation on these weights
        # Therefore not GPU heavy
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Converting size to 512
        backbone_out_dim = self.backbone.num_features  # usually 768 for ViT-Base

        if backbone_out_dim != output_dim:
            self.projection = nn.Linear(backbone_out_dim, output_dim)
        else:
            self.projection = nn.Identity()

        print(f"VisionModule ready: output shape = [Batch, {output_dim}]")

    def forward(self, x):
        """
        x: Tensor of shape [Batch, 3, H, W]
        """
        features = self.backbone(x)      # [Batch, 768]
        out = self.projection(features)  # [Batch, 512]
        return out
