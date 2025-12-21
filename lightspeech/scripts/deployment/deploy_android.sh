#!/bin/bash
################################################################################
# Android Deployment Script
# 
# Converts emotion recognition model to TensorFlow Lite format and
# prepares it for Android deployment with Android Studio integration.
#
# Usage:
#   bash scripts/deployment/deploy_android.sh
#
# Prerequisites:
#   - TensorFlow installed (pip install tensorflow)
#   - ONNX-TF installed (pip install onnx-tf)
#   - Android Studio (optional, for app integration)
################################################################################

set -e  # Exit on error

# Configuration
MODEL_PATH="results/models/models_quantized.pt"
OUTPUT_DIR="deployment/android"
TFLITE_MODEL="emotion_model.tflite"
TFLITE_QUANTIZED="emotion_model_quantized.tflite"

echo "================================"
echo "Android Deployment Script"
echo "================================"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found: $MODEL_PATH"
    echo "[INFO] Please run quantization first:"
    echo "  python scripts/03_compress_model.py --quantize"
    exit 1
fi

# Create output directory
echo ""
echo "[1/4] Creating output directory..."
mkdir -p "$OUTPUT_DIR"/{models,assets,app}

# Check Python dependencies
echo ""
echo "[2/4] Checking dependencies..."
python3 -c "import tensorflow; print('[INFO] TensorFlow version:', tensorflow.__version__)" || {
    echo "[ERROR] TensorFlow not installed"
    echo "[INFO] Install with: pip install tensorflow"
    exit 1
}

# Convert model to TFLite
echo ""
echo "[3/4] Converting model to TensorFlow Lite..."

# Create conversion script
cat > "$OUTPUT_DIR/convert_to_tflite.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Convert PyTorch model to TensorFlow Lite."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from lightspeech.code.deployment.tflite_converter import TFLiteConverter

# Model paths
MODEL_PATH = "results/models/models_quantized.pt"
OUTPUT_DIR = "deployment/android/models"

print("[INFO] Loading PyTorch model...")
try:
    # Load model architecture (adjust based on your model)
    from lightspeech.code.models.cnn import StudentCNN
    
    model = StudentCNN(num_classes=8)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("[INFO] Model loaded successfully")
    
    # Convert to TFLite
    converter = TFLiteConverter(model, input_shape=(1, 1, 128, 300))
    
    # Regular TFLite
    print("\n[INFO] Converting to TFLite (float32)...")
    tflite_path = os.path.join(OUTPUT_DIR, "emotion_model.tflite")
    converter.convert(tflite_path, quantize=False, optimization='default')
    
    # Quantized TFLite
    print("\n[INFO] Converting to TFLite (int8 quantized)...")
    tflite_quantized_path = os.path.join(OUTPUT_DIR, "emotion_model_quantized.tflite")
    converter.convert(tflite_quantized_path, quantize=True, optimization='size')
    
    # Validate
    print("\n[INFO] Validating conversions...")
    converter.tflite_path = tflite_path
    converter.validate(tolerance=1e-3)
    
    converter.tflite_path = tflite_quantized_path
    converter.validate(tolerance=1e-2)
    
    print("\n[SUCCESS] Conversion complete!")
    print(f"Models saved to: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"[ERROR] Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

# Run conversion
python3 "$OUTPUT_DIR/convert_to_tflite.py"

# Create Android integration guide
echo ""
echo "[4/4] Creating Android integration guide..."

cat > "$OUTPUT_DIR/ANDROID_INTEGRATION.md" << 'MARKDOWN'
# Android Integration Guide

## Model Files

- `emotion_model.tflite` - Float32 model (~5-10 MB)
- `emotion_model_quantized.tflite` - INT8 quantized model (~2-5 MB)

Use the quantized model for better performance on mobile devices.

## Integration Steps

### 1. Add Model to Android Project

Copy the TFLite model to your Android project:
```bash
cp deployment/android/models/emotion_model_quantized.tflite \
   YourAndroidApp/app/src/main/assets/
```

### 2. Add TensorFlow Lite Dependency

In your `app/build.gradle`:
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### 3. Load Model in Kotlin/Java

**Kotlin:**
```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class EmotionRecognizer(private val context: Context) {
    private lateinit var interpreter: Interpreter
    
    private val emotions = arrayOf(
        "angry", "disgust", "fear", "happy", 
        "neutral", "sad", "surprise", "calm"
    )
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        val model = loadModelFile("emotion_model_quantized.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(model, options)
    }
    
    private fun loadModelFile(filename: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(melSpectrogram: Array<Array<FloatArray>>): Pair<String, Float> {
        // Input: melSpectrogram [1][128][300]
        // Output: logits [1][8]
        
        val inputBuffer = ByteBuffer.allocateDirect(1 * 1 * 128 * 300 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // Fill input buffer
        for (i in 0 until 128) {
            for (j in 0 until 300) {
                inputBuffer.putFloat(melSpectrogram[0][i][j])
            }
        }
        
        val outputBuffer = ByteBuffer.allocateDirect(1 * 8 * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Parse output
        outputBuffer.rewind()
        val logits = FloatArray(8)
        for (i in 0 until 8) {
            logits[i] = outputBuffer.float
        }
        
        // Get prediction
        val maxIdx = logits.indices.maxByOrNull { logits[it] } ?: 0
        val confidence = softmax(logits)[maxIdx]
        
        return Pair(emotions[maxIdx], confidence)
    }
    
    private fun softmax(logits: FloatArray): FloatArray {
        val expValues = logits.map { exp(it.toDouble()).toFloat() }
        val sum = expValues.sum()
        return expValues.map { it / sum }.toFloatArray()
    }
    
    fun close() {
        interpreter.close()
    }
}
```

**Java:**
```java
import org.tensorflow.lite.Interpreter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class EmotionRecognizer {
    private Interpreter interpreter;
    private String[] emotions = {
        "angry", "disgust", "fear", "happy",
        "neutral", "sad", "surprise", "calm"
    };
    
    public EmotionRecognizer(Context context) {
        loadModel(context);
    }
    
    private void loadModel(Context context) {
        try {
            ByteBuffer model = loadModelFile(context, "emotion_model_quantized.tflite");
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            interpreter = new Interpreter(model, options);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public Pair<String, Float> predict(float[][][] melSpectrogram) {
        // Input: melSpectrogram [1][128][300]
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * 1 * 128 * 300 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());
        
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 300; j++) {
                inputBuffer.putFloat(melSpectrogram[0][i][j]);
            }
        }
        
        float[][] output = new float[1][8];
        interpreter.run(inputBuffer, output);
        
        // Find max
        int maxIdx = 0;
        float maxVal = output[0][0];
        for (int i = 1; i < 8; i++) {
            if (output[0][i] > maxVal) {
                maxVal = output[0][i];
                maxIdx = i;
            }
        }
        
        return new Pair<>(emotions[maxIdx], maxVal);
    }
}
```

### 4. Audio Preprocessing

You'll need to convert audio to mel spectrogram (128 x 300):

```kotlin
// Use librosa-equivalent Android library or implement DSP
// Recommended: TarsosDSP or custom FFT implementation
```

### 5. Permissions

Add to `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

## Model Specifications

- **Input Shape**: `[1, 1, 128, 300]`
  - Batch size: 1
  - Channels: 1 (mono)
  - Height: 128 (mel bins)
  - Width: 300 (time frames)

- **Output Shape**: `[1, 8]`
  - Logits for 8 emotion classes

- **Preprocessing**:
  1. Load audio at 16kHz
  2. Extract mel spectrogram (128 bins, 300 frames)
  3. Normalize: `(spec - mean) / std`
  4. Convert to float32

- **Postprocessing**:
  - Apply softmax to logits
  - Select max probability as prediction

## Performance Tips

1. **Use quantized model** for faster inference
2. **Set thread count** based on device cores
3. **Use NNAPI** delegate for hardware acceleration:
   ```kotlin
   options.addDelegate(NnApiDelegate())
   ```
4. **GPU acceleration** (if available):
   ```kotlin
   options.addDelegate(GpuDelegate())
   ```

## Testing

Test the model with sample audio files before deploying:
```kotlin
val recognizer = EmotionRecognizer(context)
val result = recognizer.predict(melSpectrogram)
Log.d("Emotion", "Predicted: ${result.first} (${result.second})")
```

## Troubleshooting

- **Model not found**: Check assets folder path
- **Slow inference**: Use quantized model + NNAPI/GPU delegate
- **Incorrect predictions**: Verify audio preprocessing matches training

## References

- [TensorFlow Lite Android](https://www.tensorflow.org/lite/guide/android)
- [TF Lite Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)
MARKDOWN

echo ""
echo "================================"
echo "Deployment Complete!"
echo "================================"
echo ""
echo "TFLite models created:"
echo "  - $OUTPUT_DIR/models/emotion_model.tflite (float32)"
echo "  - $OUTPUT_DIR/models/emotion_model_quantized.tflite (int8)"
echo ""
echo "Integration guide:"
echo "  - $OUTPUT_DIR/ANDROID_INTEGRATION.md"
echo ""
echo "Next steps:"
echo "  1. Copy TFLite model to Android project assets/"
echo "  2. Follow integration guide in ANDROID_INTEGRATION.md"
echo "  3. Test with sample audio files"
echo ""
