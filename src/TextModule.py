import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextModule(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the text encoder
        
        Args:
            model_name: Pre-trained model from Hugging Face
                       (MiniLM outputs 384-dim vectors by default)
        """
        super(TextModule, self).__init__()
        
        # Load pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze weights (no training, just feature extraction)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()  # Set to evaluation mode
    
    def forward(self, text_list):
        """
        Convert text to vectors
        
        Args:
            text_list: List of strings, e.g., ["cat", "dog running"]
        
        Returns:
            Tensor of shape [Batch, 384]
        """
        # Tokenize (convert words to numbers)
        encoded = self.tokenizer(
            text_list,
            padding=True,        # Make all sequences same length
            truncation=True,     # Cut off if too long
            return_tensors='pt'  # Return PyTorch tensors
        )
        
        # FIX: Move tokenizer output to the same device as the model
        device = next(self.model.parameters()).device
        encoded = {key: val.to(device) for key, val in encoded.items()}
        
        # Get embeddings from model
        with torch.no_grad():  # Don't compute gradients (faster)
            outputs = self.model(**encoded)
        
        # Use [CLS] token embedding (standard practice for sentence vectors)
        # Shape: [Batch, 384]
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings