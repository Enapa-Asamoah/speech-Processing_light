#!/bin/bash
################################################################################
# Raspberry Pi Deployment Script
# 
# Deploys emotion recognition model to Raspberry Pi for edge inference.
# Converts model to ONNX format and sets up inference environment.
#
# Usage:
#   bash scripts/deployment/deploy_raspberry_pi.sh
#
# Prerequisites:
#   - Raspberry Pi with Raspberry Pi OS (64-bit recommended)
#   - SSH access configured
#   - Python 3.8+ on Raspberry Pi
################################################################################

set -e  # Exit on error

# Configuration
PI_USER="pi"
PI_HOST="raspberrypi.local"  # Change to your Pi's hostname or IP
PI_DIR="/home/pi/emotion_recognition"
MODEL_PATH="results/models/models_quantized.onnx"
LOCAL_INFERENCE_SCRIPT="scripts/deployment/inference_raspberry_pi.py"

echo "================================"
echo "Raspberry Pi Deployment Script"
echo "================================"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found: $MODEL_PATH"
    echo "[INFO] Please run quantization first:"
    echo "  python scripts/03_compress_model.py --quantize"
    exit 1
fi

# Test SSH connection
echo ""
echo "[1/5] Testing SSH connection to $PI_USER@$PI_HOST..."
if ! ssh -o ConnectTimeout=5 "$PI_USER@$PI_HOST" "echo 'SSH connection successful'"; then
    echo "[ERROR] Cannot connect to Raspberry Pi"
    echo "[INFO] Check SSH connection: ssh $PI_USER@$PI_HOST"
    exit 1
fi

# Create directory on Pi
echo ""
echo "[2/5] Creating directory on Raspberry Pi..."
ssh "$PI_USER@$PI_HOST" "mkdir -p $PI_DIR/{models,data,scripts}"

# Install dependencies on Pi
echo ""
echo "[3/5] Installing dependencies on Raspberry Pi..."
ssh "$PI_USER@$PI_HOST" << 'EOF'
    # Update package list
    sudo apt-get update -qq
    
    # Install system dependencies
    sudo apt-get install -y python3-pip python3-numpy libopenblas-dev
    
    # Install Python packages
    pip3 install --upgrade pip
    pip3 install onnxruntime librosa soundfile numpy pillow
    
    echo "[INFO] Dependencies installed successfully"
EOF

# Copy model to Pi
echo ""
echo "[4/5] Copying model to Raspberry Pi..."
scp "$MODEL_PATH" "$PI_USER@$PI_HOST:$PI_DIR/models/"
echo "[INFO] Model copied successfully"

# Copy inference script to Pi
echo ""
echo "[5/5] Copying inference script to Raspberry Pi..."

# Create inference script if it doesn't exist
if [ ! -f "$LOCAL_INFERENCE_SCRIPT" ]; then
    echo "[INFO] Creating inference script..."
    cat > "$LOCAL_INFERENCE_SCRIPT" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Raspberry Pi Inference Script
Runs emotion recognition on Raspberry Pi using ONNX Runtime.
"""

import os
import sys
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "calm"]

def load_model(model_path):
    """Load ONNX model."""
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path)
    print(f"Model loaded successfully")
    return session

def preprocess_audio(audio_path):
    """Preprocess audio file to model input format."""
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa not installed. Install with: pip3 install librosa")
        sys.exit(1)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # Resize to fixed shape (128, 300)
    if mel_spec_db.shape[1] < 300:
        pad_width = 300 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :300]
    
    # Add batch and channel dimensions
    mel_spec_db = mel_spec_db[np.newaxis, np.newaxis, :, :]
    
    return mel_spec_db.astype(np.float32)

def predict(session, audio_path):
    """Run inference on audio file."""
    # Preprocess
    input_data = preprocess_audio(audio_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    latency = (time.time() - start_time) * 1000  # milliseconds
    
    # Get prediction
    logits = outputs[0][0]
    pred_idx = np.argmax(logits)
    confidence = np.exp(logits[pred_idx]) / np.sum(np.exp(logits))
    
    return EMOTIONS[pred_idx], confidence, latency

def main():
    """Main inference loop."""
    # Paths
    model_path = "models/models_quantized.onnx"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    # Load model
    session = load_model(model_path)
    
    print("\n=== Raspberry Pi Emotion Recognition ===")
    print("Model ready for inference")
    print("\nUsage:")
    print("  python3 scripts/inference_raspberry_pi.py <audio_file.wav>")
    print("\nOr run in interactive mode:")
    
    # Check for command line audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file not found: {audio_path}")
            sys.exit(1)
        
        print(f"\nProcessing: {audio_path}")
        emotion, confidence, latency = predict(session, audio_path)
        
        print(f"\nPrediction: {emotion}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Latency: {latency:.2f} ms")
    else:
        # Interactive mode
        print("\n[INFO] Enter audio file path (or 'quit' to exit):")
        while True:
            try:
                audio_path = input("> ").strip()
                
                if audio_path.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break
                
                if not os.path.exists(audio_path):
                    print(f"ERROR: File not found: {audio_path}")
                    continue
                
                emotion, confidence, latency = predict(session, audio_path)
                print(f"Prediction: {emotion} ({confidence:.2%}) | Latency: {latency:.2f} ms")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT
    chmod +x "$LOCAL_INFERENCE_SCRIPT"
fi

scp "$LOCAL_INFERENCE_SCRIPT" "$PI_USER@$PI_HOST:$PI_DIR/scripts/inference_raspberry_pi.py"
echo "[INFO] Inference script copied successfully"

# Run test inference on Pi
echo ""
echo "================================"
echo "Deployment Complete!"
echo "================================"
echo ""
echo "Model deployed to: $PI_HOST:$PI_DIR"
echo ""
echo "To run inference on Raspberry Pi:"
echo "  1. SSH to Pi: ssh $PI_USER@$PI_HOST"
echo "  2. Navigate to: cd $PI_DIR"
echo "  3. Run inference:"
echo "     python3 scripts/inference_raspberry_pi.py <audio_file.wav>"
echo ""
echo "Example:"
echo "  python3 scripts/inference_raspberry_pi.py data/test_audio.wav"
echo ""

# Optional: Run benchmark on Pi
read -p "Run benchmark on Raspberry Pi? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running benchmark on Raspberry Pi..."
    ssh "$PI_USER@$PI_HOST" "cd $PI_DIR && python3 -c '
import time
import numpy as np
import onnxruntime as ort

model_path = \"models/models_quantized.onnx\"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Generate random input
dummy_input = np.random.randn(1, 1, 128, 300).astype(np.float32)

# Warmup
for _ in range(10):
    session.run(None, {input_name: dummy_input})

# Benchmark
num_runs = 100
start = time.time()
for _ in range(num_runs):
    session.run(None, {input_name: dummy_input})
end = time.time()

avg_latency = (end - start) / num_runs * 1000
print(f\"Average latency: {avg_latency:.2f} ms\")
print(f\"Throughput: {1000/avg_latency:.2f} inferences/sec\")
'"
fi

echo ""
echo "Deployment complete!"
