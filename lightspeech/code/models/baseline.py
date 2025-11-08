"""
Baseline emotion recognition model
"""

import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """
    Baseline CNN-based emotion recognition model.
    
    Students can modify this architecture or create alternatives
    (e.g., Transformer-based, CNN-LSTM hybrid).
    """
    
    def __init__(self, config):
        super(BaselineModel, self).__init__()
        
        # TODO: Implement model architecture based on config
        # Example structure:
        # - Input: Log-Mel spectrogram (128 bins × 300 frames)
        # - CNN layers for feature extraction
        # - Global pooling
        # - Fully connected layers
        # - Output: Emotion classification
        
        self.config = config
        # Placeholder - students should implement
        raise NotImplementedError("Students should implement the baseline model architecture")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, channels, height, width)
        
        Returns:
            Emotion predictions (batch_size, num_emotions)
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Students should implement the forward pass")

