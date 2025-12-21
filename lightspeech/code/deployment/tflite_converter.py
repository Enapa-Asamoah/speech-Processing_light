"""
TensorFlow Lite Conversion Utilities

Converts PyTorch models to TensorFlow Lite format for mobile/edge deployment.
Supports quantization and optimization for Android and iOS devices.

Usage:
    from lightspeech.code.deployment.tflite_converter import TFLiteConverter
    
    converter = TFLiteConverter(model, input_shape=(1, 1, 128, 300))
    converter.convert("model.tflite", quantize=True)
"""

import os
import torch
import numpy as np
from pathlib import Path
import tempfile


class TFLiteConverter:
    """Convert PyTorch models to TensorFlow Lite format."""
    
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
        self.tflite_path = None
        
    def convert(self, output_path, quantize=False, optimization='default'):
        """
        Convert PyTorch model to TensorFlow Lite.
        
        Args:
            output_path: Path to save TFLite model
            quantize: Whether to apply INT8 quantization
            optimization: Optimization mode ('default', 'size', 'latency')
        
        Returns:
            Path to converted TFLite model
        """
        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
        except ImportError:
            raise ImportError(
                "TensorFlow and ONNX-TF required for TFLite conversion.\n"
                "Install with: pip install tensorflow onnx onnx-tf"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Converting model to TensorFlow Lite: {output_path}")
        print(f"[INFO] Input shape: {self.input_shape}")
        
        # Step 1: PyTorch -> ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "temp_model.onnx"
            
            dummy_input = torch.randn(self.input_shape).to(self.device)
            
            print("[INFO] Step 1/3: Converting PyTorch to ONNX...")
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )
            
            # Step 2: ONNX -> TensorFlow SavedModel
            print("[INFO] Step 2/3: Converting ONNX to TensorFlow...")
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            
            saved_model_path = Path(tmpdir) / "saved_model"
            tf_rep.export_graph(str(saved_model_path))
            
            # Step 3: TensorFlow SavedModel -> TFLite
            print("[INFO] Step 3/3: Converting TensorFlow to TFLite...")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            
            # Set optimization flags
            if optimization == 'size':
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            elif optimization == 'latency':
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Apply quantization if requested
            if quantize:
                print("[INFO] Applying INT8 quantization...")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                
                # Representative dataset for quantization calibration
                def representative_dataset():
                    for _ in range(100):
                        data = np.random.randn(*self.input_shape).astype(np.float32)
                        yield [data]
                
                converter.representative_dataset = representative_dataset
            
            # Convert
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
        
        self.tflite_path = str(output_path)
        
        # Get model size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] TFLite model saved: {output_path} ({size_mb:.2f} MB)")
        
        return str(output_path)
    
    def validate(self, tolerance=1e-2):
        """
        Validate TFLite model against PyTorch model.
        
        Args:
            tolerance: Numerical tolerance for comparison (higher for quantized)
        """
        if self.tflite_path is None:
            raise ValueError("No TFLite model converted yet. Call convert() first.")
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for validation")
        
        print(f"[INFO] Validating TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create test input
        test_input = torch.randn(self.input_shape).to(self.device)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = self.model(test_input).cpu().numpy()
        
        # Get TFLite output
        interpreter.set_tensor(input_details[0]['index'], test_input.cpu().numpy())
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Compare outputs
        max_diff = np.abs(pytorch_output - tflite_output).max()
        mean_diff = np.abs(pytorch_output - tflite_output).mean()
        
        print(f"[INFO] Max difference: {max_diff:.6f}")
        print(f"[INFO] Mean difference: {mean_diff:.6f}")
        
        if max_diff < tolerance:
            print(f"[SUCCESS] Validation passed (tolerance: {tolerance})")
            return True
        else:
            print(f"[WARNING] Validation failed (max diff {max_diff} > tolerance {tolerance})")
            print("[INFO] Higher tolerance may be acceptable for quantized models")
            return False
    
    def benchmark(self, num_runs=100):
        """
        Benchmark TFLite model inference speed.
        
        Args:
            num_runs: Number of inference runs for averaging
        """
        if self.tflite_path is None:
            raise ValueError("No TFLite model converted yet. Call convert() first.")
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for benchmarking")
        
        import time
        
        print(f"[INFO] Benchmarking TFLite model ({num_runs} runs)...")
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create test input
        test_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        end = time.time()
        
        avg_latency = (end - start) / num_runs * 1000  # milliseconds
        print(f"[INFO] Average latency: {avg_latency:.2f} ms")
        print(f"[INFO] Throughput: {1000/avg_latency:.2f} inferences/sec")
        
        return avg_latency
    
    def get_model_info(self):
        """Get detailed information about TFLite model."""
        if self.tflite_path is None:
            raise ValueError("No TFLite model converted yet. Call convert() first.")
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required")
        
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n=== TFLite Model Info ===")
        print(f"\nInput Details:")
        print(f"  Name: {input_details[0]['name']}")
        print(f"  Shape: {input_details[0]['shape']}")
        print(f"  Dtype: {input_details[0]['dtype']}")
        
        print(f"\nOutput Details:")
        print(f"  Name: {output_details[0]['name']}")
        print(f"  Shape: {output_details[0]['shape']}")
        print(f"  Dtype: {output_details[0]['dtype']}")
        
        size_mb = Path(self.tflite_path).stat().st_size / (1024 * 1024)
        print(f"\nModel Size: {size_mb:.2f} MB")
        print("========================\n")
        
        return {
            'input_details': input_details,
            'output_details': output_details,
            'size_mb': size_mb
        }


def quick_convert(model, output_path, input_shape=(1, 1, 128, 300), 
                  quantize=False, validate=True, benchmark=False):
    """
    High-level function for quick PyTorch to TFLite conversion.
    
    Args:
        model: PyTorch model
        output_path: Output TFLite path
        input_shape: Input tensor shape
        quantize: Apply INT8 quantization
        validate: Validate conversion
        benchmark: Run benchmark
    
    Returns:
        Path to converted model
    """
    converter = TFLiteConverter(model, input_shape)
    
    # Convert
    tflite_path = converter.convert(output_path, quantize=quantize)
    
    # Validate
    if validate:
        converter.validate(tolerance=1e-2 if quantize else 1e-3)
    
    # Benchmark
    if benchmark:
        converter.benchmark()
    
    # Show info
    converter.get_model_info()
    
    return tflite_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite")
    parser.add_argument("--model", required=True, help="Path to PyTorch model")
    parser.add_argument("--output", required=True, help="Output TFLite path")
    parser.add_argument("--input_shape", nargs='+', type=int,
                        default=[1, 1, 128, 300], help="Input shape")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark inference")
    
    args = parser.parse_args()
    
    print("Note: Model architecture must be loaded separately.")
    print("Use TFLiteConverter class in your script instead.")
