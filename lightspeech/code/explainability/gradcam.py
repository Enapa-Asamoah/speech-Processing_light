"""
Grad-CAM implementation for 1D and 2D CNNs to explain emotional cues in speech.

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of the input
are important for the model's prediction by computing gradients of the target class
with respect to feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple


class GradCAM:
    """
    Grad-CAM for CNNs (supports both 1D and 2D convolutions).
    
    Args:
        model: The neural network model
        target_layer: The layer to compute Grad-CAM on (typically last conv layer)
        device: Device to run computations on
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = "cpu"):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        """Save the forward pass activations."""
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_input, grad_output):
        """Save the gradients during backward pass."""
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor (B, C, H, W) for 2D or (B, C, L) for 1D
            target_class: Target class for visualization. If None, uses predicted class.
            
        Returns:
            cam: Grad-CAM heatmap (H, W) for 2D or (L,) for 1D
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W) or (C, L)
        activations = self.activations[0]  # (C, H, W) or (C, L)
        
        # Global average pooling of gradients (weights)
        if gradients.dim() == 3:  # 2D case: (C, H, W)
            weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
            cam = (weights * activations).sum(dim=0)  # (H, W)
        elif gradients.dim() == 2:  # 1D case: (C, L)
            weights = gradients.mean(dim=1, keepdim=True)  # (C, 1)
            cam = (weights * activations).sum(dim=0)  # (L,)
        else:
            raise ValueError(f"Unexpected gradient dimensions: {gradients.dim()}")
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def remove_hooks(self):
        """Remove the registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        try:
            self.remove_hooks()
        except:
            pass


class GuidedBackprop:
    """
    Guided Backpropagation for visualizing what the network learned.
    Works with both 1D and 2D CNNs.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.gradients = None
        
        # Store original relu backward functions
        self.relu_outputs = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for ReLU layers."""
        def relu_hook_function(module, grad_in, grad_out):
            """Guided backprop ReLU hook."""
            return (F.relu(grad_in[0]),)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_hook_function)
    
    def generate_gradients(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate guided backpropagation gradients.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class. If None, uses predicted class.
            
        Returns:
            gradients: Guided backprop gradients
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        gradients = input_tensor.grad[0].cpu().numpy()
        
        return gradients


def visualize_gradcam_2d(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize Grad-CAM heatmap overlaid on 2D image.
    
    Args:
        image: Original image (H, W) or (H, W, C)
        cam: Grad-CAM heatmap (H, W)
        alpha: Transparency of overlay
        colormap: Matplotlib colormap
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(2)
    
    axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[0].set_title("Original Spectrogram")
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    from scipy.ndimage import zoom
    if cam.shape != image.shape[:2]:
        zoom_factor = (image.shape[0] / cam.shape[0], image.shape[1] / cam.shape[1])
        cam = zoom(cam, zoom_factor, order=1)
    
    axes[1].imshow(cam, cmap=colormap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[2].imshow(cam, cmap=colormap, alpha=alpha)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


def visualize_gradcam_1d(
    signal: np.ndarray,
    cam: np.ndarray,
    emotion_label: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize Grad-CAM for 1D signal (time series).
    
    Args:
        signal: Original 1D signal (L,) or (C, L)
        cam: Grad-CAM heatmap (L,)
        emotion_label: Emotion label for title
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Handle multi-channel input
    if signal.ndim == 2:
        signal = signal.mean(axis=0)  # Average across channels
    
    time_steps = np.arange(len(signal))
    
    # Original signal
    axes[0].plot(time_steps, signal, 'b-', linewidth=0.8)
    axes[0].set_title("Original Audio Signal")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # Grad-CAM heatmap
    from scipy.ndimage import zoom
    if len(cam) != len(signal):
        zoom_factor = len(signal) / len(cam)
        cam = zoom(cam, zoom_factor, order=1)
    
    axes[1].plot(time_steps, cam, 'r-', linewidth=1.5)
    axes[1].fill_between(time_steps, cam, alpha=0.3, color='red')
    axes[1].set_title("Grad-CAM Attention Map")
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Attention Weight")
    axes[1].grid(True, alpha=0.3)
    
    # Signal with highlighted regions
    axes[2].plot(time_steps, signal, 'b-', linewidth=0.8, alpha=0.5)
    
    # Overlay gradient as color intensity
    scatter = axes[2].scatter(time_steps, signal, c=cam, cmap='YlOrRd', 
                             s=20, alpha=0.7, edgecolors='none')
    axes[2].set_title(f"Signal with Attention Overlay" + 
                     (f" - {emotion_label}" if emotion_label else ""))
    axes[2].set_xlabel("Time Steps")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=axes[2], label='Attention')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


def get_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    """
    Get target layer for Grad-CAM visualization.
    
    Args:
        model: Neural network model
        layer_name: Name of target layer. If None, returns last conv layer.
        
    Returns:
        target_layer: Target layer module
    """
    if layer_name:
        # Try to get layer by name
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Find last convolutional layer
    conv_layers = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_layers.append(module)
    
    if not conv_layers:
        raise ValueError("No convolutional layers found in model")
    
    return conv_layers[-1]


def explain_emotion_prediction(
    model: nn.Module,
    input_tensor: torch.Tensor,
    emotion_labels: List[str],
    target_layer: Optional[nn.Module] = None,
    device: str = "cpu",
    save_dir: Optional[str] = None
) -> dict:
    """
    Generate comprehensive explanation for emotion prediction.
    
    Args:
        model: Trained emotion recognition model
        input_tensor: Input tensor (spectrogram or audio features)
        emotion_labels: List of emotion label names
        target_layer: Layer to compute Grad-CAM on
        device: Device to run on
        save_dir: Directory to save visualizations
        
    Returns:
        results: Dictionary containing predictions, Grad-CAM, and visualizations
    """
    import os
    
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
        confidence = probs[pred_class].item()
    
    # Get target layer if not provided
    if target_layer is None:
        target_layer = get_target_layer(model)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer, device)
    cam = gradcam.generate_cam(input_tensor, target_class=pred_class)
    
    # Prepare results
    results = {
        'predicted_class': pred_class,
        'predicted_emotion': emotion_labels[pred_class],
        'confidence': confidence,
        'all_probabilities': {emotion_labels[i]: probs[i].item() 
                             for i in range(len(emotion_labels))},
        'gradcam': cam,
        'input_shape': input_tensor.shape
    }
    
    # Generate visualizations
    input_np = input_tensor[0].cpu().numpy()
    
    # Check if 1D or 2D
    is_2d = input_np.ndim == 3 or (input_np.ndim == 2 and input_np.shape[0] <= 3)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 
                                f"gradcam_{emotion_labels[pred_class]}_{confidence:.2f}.png")
        
        if is_2d and cam.ndim == 2:
            fig = visualize_gradcam_2d(input_np, cam, save_path=save_path)
        else:
            fig = visualize_gradcam_1d(input_np, cam, 
                                       emotion_label=emotion_labels[pred_class],
                                       save_path=save_path)
        results['figure'] = fig
        results['save_path'] = save_path
    
    # Cleanup
    gradcam.remove_hooks()
    
    return results


def batch_explain_emotions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    emotion_labels: List[str],
    num_samples: int = 10,
    device: str = "cpu",
    save_dir: Optional[str] = None
) -> List[dict]:
    """
    Generate explanations for multiple samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        emotion_labels: List of emotion labels
        num_samples: Number of samples to explain
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        all_results: List of explanation dictionaries
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    target_layer = get_target_layer(model)
    
    for i, (inputs, labels) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        for j in range(min(inputs.shape[0], num_samples - i)):
            sample_input = inputs[j:j+1]
            true_label = labels[j].item()
            
            sample_save_dir = os.path.join(save_dir, f"sample_{i}_{j}") if save_dir else None
            
            results = explain_emotion_prediction(
                model, sample_input, emotion_labels,
                target_layer=target_layer,
                device=device,
                save_dir=sample_save_dir
            )
            
            results['true_emotion'] = emotion_labels[true_label]
            results['correct'] = (results['predicted_class'] == true_label)
            
            all_results.append(results)
            
            if len(all_results) >= num_samples:
                break
    
    return all_results
