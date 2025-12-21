"""
ONNX Model Conversion Utilities

Converts PyTorch models to ONNX format for cross-platform deployment.
Supports optimization and quantization for edge devices.

Usage:
    from lightspeech.code.deployment.onnx_converter import ONNXConverter
    
    converter = ONNXConverter(model, input_shape=(1, 1, 128, 300))
    converter.export("model.onnx")
    converter.validate()
"""

import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
from onnx import optimizer
import numpy as np
from pathlib import Path


class ONNXConverter:
    """Convert PyTorch models to optimized ONNX format."""
    
    def __init__(self, model, input_shape=(1, 1, 128, 300), device='cpu'):
        """
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape (batch, channels, height, width)
            device: Device for model ('cpu' or 'cuda')
        """
        self.model = model.to(device).eval()
        self.input_shape = input_shape
        self.device = device
        self.onnx_path = None
        
    def export(self, output_path, opset_version=14, dynamic_axes=None):
        """
        Export PyTorch model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version (14 for better quantization support)
            dynamic_axes: Dict of dynamic axes for variable input sizes
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Default dynamic axes for variable batch size
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        print(f"[INFO] Exporting model to ONNX: {output_path}")
        print(f"[INFO] Input shape: {self.input_shape}")
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        self.onnx_path = str(output_path)
        print(f"[SUCCESS] Model exported to {output_path}")
        
        # Get model size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] Model size: {size_mb:.2f} MB")
        
        return str(output_path)
    
    def optimize(self, output_path=None):
        """
        Optimize ONNX model using ONNX optimizer.
        
        Args:
            output_path: Path to save optimized model (overwrites original if None)
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model exported yet. Call export() first.")
        
        print(f"[INFO] Optimizing ONNX model...")
        
        # Load model
        model = onnx.load(self.onnx_path)
        
        # Apply optimization passes
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'fuse_consecutive_transposes',
            'fuse_consecutive_squeezes',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
        ]
        
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        if output_path is None:
            output_path = self.onnx_path
        
        onnx.save(optimized_model, output_path)
        self.onnx_path = output_path
        
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Optimized model saved: {output_path} ({size_mb:.2f} MB)")
        
        return output_path
    
    def quantize(self, output_path, quantization_mode='dynamic'):
        """
        Quantize ONNX model to INT8.
        
        Args:
            output_path: Path to save quantized model
            quantization_mode: 'dynamic' or 'static' quantization
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model exported yet. Call export() first.")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
            from onnxruntime.quantization import CalibrationDataReader
        except ImportError:
            raise ImportError("onnxruntime quantization not available. Install with: pip install onnxruntime")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Quantizing model ({quantization_mode})...")
        
        if quantization_mode == 'dynamic':
            quantize_dynamic(
                self.onnx_path,
                str(output_path),
                weight_type=QuantType.QUInt8
            )
        else:
            raise NotImplementedError("Static quantization requires calibration data reader")
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Quantized model saved: {output_path} ({size_mb:.2f} MB)")
        
        return str(output_path)
    
    def validate(self, pytorch_output=None, tolerance=1e-3):
        """
        Validate ONNX model against PyTorch model.
        
        Args:
            pytorch_output: Expected output from PyTorch model (computed if None)
            tolerance: Numerical tolerance for comparison
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model exported yet. Call export() first.")
        
        print(f"[INFO] Validating ONNX model...")
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(self.onnx_path)
        
        # Create test input
        test_input = torch.randn(self.input_shape).to(self.device)
        
        # Get PyTorch output
        if pytorch_output is None:
            with torch.no_grad():
                pytorch_output = self.model(test_input).cpu().numpy()
        
        # Get ONNX output
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()
        
        print(f"[INFO] Max difference: {max_diff:.6f}")
        print(f"[INFO] Mean difference: {mean_diff:.6f}")
        
        if max_diff < tolerance:
            print(f"[SUCCESS] Validation passed (tolerance: {tolerance})")
            return True
        else:
            print(f"[WARNING] Validation failed (max diff {max_diff} > tolerance {tolerance})")
            return False
    
    def benchmark(self, num_runs=100):
        """
        Benchmark ONNX model inference speed.
        
        Args:
            num_runs: Number of inference runs for averaging
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model exported yet. Call export() first.")
        
        import time
        
        print(f"[INFO] Benchmarking ONNX model ({num_runs} runs)...")
        
        # Create session
        ort_session = ort.InferenceSession(self.onnx_path)
        
        # Create test input
        test_input = torch.randn(self.input_shape).cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        
        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, ort_inputs)
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = ort_session.run(None, ort_inputs)
        end = time.time()
        
        avg_latency = (end - start) / num_runs * 1000  # milliseconds
        print(f"[INFO] Average latency: {avg_latency:.2f} ms")
        print(f"[INFO] Throughput: {1000/avg_latency:.2f} inferences/sec")
        
        return avg_latency


def convert_model(model_path, output_path, input_shape=(1, 1, 128, 300),
                  optimize=True, quantize=False, validate=True):
    """
    High-level function to convert PyTorch model to ONNX.
    
    Args:
        model_path: Path to PyTorch model checkpoint (.pt or .pth)
        output_path: Path to save ONNX model
        input_shape: Input tensor shape
        optimize: Whether to optimize ONNX model
        quantize: Whether to quantize to INT8
        validate: Whether to validate outputs
    
    Returns:
        Path to final ONNX model
    """
    # Load PyTorch model
    print(f"[INFO] Loading PyTorch model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Need to instantiate model architecture (user must provide)
    raise NotImplementedError(
        "Model architecture must be instantiated. Use ONNXConverter class directly."
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--model", required=True, help="Path to PyTorch model")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--input_shape", nargs='+', type=int, 
                        default=[1, 1, 128, 300], help="Input shape")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")
    parser.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark inference")
    
    args = parser.parse_args()
    
    print("Note: Model architecture must be loaded separately.")
    print("Use ONNXConverter class in your script instead.")
