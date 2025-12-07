# export_onnx.py
import argparse
import torch
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
# ensure project root is importable when running the script directly
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from lightspeech.code.models.baseline import get_model  # adjust to your model factory
# load the model architecture and weights
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='Dummy batch size to use when exporting ONNX')
parser.add_argument('--ckpt', type=str, default='results/models/baseline_model.pth', help='Checkpoint path')
args = parser.parse_args()

ckpt_path = args.ckpt
ckpt = torch.load(ckpt_path, map_location='cpu')
state = None
if isinstance(ckpt, dict) and ('model_state' in ckpt or 'state_dict' in ckpt):
    state = ckpt.get('model_state', ckpt.get('state_dict'))
elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
    # it already looks like a state_dict
    state = ckpt
else:
    # unknown format
    raise RuntimeError(f"Unrecognized checkpoint format: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

# strip common DataParallel prefix if present
def _strip_module_prefix(sd):
    new_sd = {}
    for k,v in sd.items():
        if k.startswith('module.'):
            new_sd[k[len('module.'):]] = v
        else:
            new_sd[k] = v
    return new_sd

state = _strip_module_prefix(state)

# Try candidate model architectures and load with strict=False to tolerate classifier size mismatch
candidates = ['mobilenet', 'cnn2d', 'transformer', 'cnn_lstm']
loaded_model = None
last_err = None
for name in candidates:
    try:
        print(f"[INFO] Trying architecture '{name}' to match checkpoint...")
        m = get_model(name=name, num_classes=8)
        try:
            m.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded checkpoint into model '{name}' (strict=False)")
            loaded_model = m
            break
        except RuntimeError as e:
            last_err = e
            print(f"[WARN] Could not load into '{name}': {e}")
    except Exception as e:
        last_err = e
        print(f"[WARN] Error instantiating model '{name}': {e}")

if loaded_model is None:
    # print some diagnostics
    print("[ERROR] Failed to match checkpoint to known architectures. Checkpoint keys:")
    for k in list(state.keys())[:50]:
        print('  ', k)
    raise RuntimeError(f"Could not load checkpoint into any candidate models. Last error: {last_err}")

model = loaded_model
model.eval()
dummy = torch.randn(args.batch_size, 3, 224, 224)  # match input shape used during training
float_export_path = 'results/models/model_float.onnx'
quant_export_path = 'results/models/models_quantized.onnx'

# Ensure output dir exists
os.makedirs(os.path.dirname(float_export_path), exist_ok=True)

used_exportdb = False
try:
    # Try ExportDB if available (torch.export)
    from torch import export as torch_export
    try:
        print('[INFO] Attempting export via torch.export (ExportDB)')
        # ExportDB API expects an arguments tuple for example inputs
        torch_export.export(model, (dummy,), float_export_path)
        print(f'[INFO] Exported via torch.export to {float_export_path}')
        used_exportdb = True
    except Exception as e:
        print(f"[WARN] torch.export failed: {e}")
        used_exportdb = False
except Exception:
    used_exportdb = False

if not used_exportdb:
    # Fall back to standard ONNX export (float) and then quantize with ONNX Runtime
    try:
        try:
            torch.onnx.export(
                model,
                dummy,
                float_export_path,
                export_params=True,
                opset_version=18,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                do_constant_folding=True
            )
            print(f"Exported float ONNX (opset 18) to {float_export_path}")
        except Exception as e:
            print(f"[WARN] float export opset 18 failed: {e}. Trying opset 13 as fallback.")
            torch.onnx.export(
                model,
                dummy,
                float_export_path,
                export_params=True,
                opset_version=13,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                do_constant_folding=True
            )
            print(f"Exported float ONNX (opset 13) to {float_export_path}")

        # Try to quantize the float ONNX with ONNX Runtime dynamic quantization
        try:
            # First run ONNX shape inference to populate/repair shape information which
            # the quantizer sometimes relies on. Save the inferred model and quantize that.
            try:
                import onnx
                print('[INFO] Running ONNX shape inference before quantization')
                m = onnx.load(float_export_path)
                m_inf = onnx.shape_inference.infer_shapes(m)
                inferred_path = float_export_path.replace('.onnx', '_inferred.onnx')
                onnx.save(m_inf, inferred_path)
                quant_target = inferred_path
                print(f'[INFO] Saved shape-inferred ONNX to {inferred_path}')
            except Exception as si_e:
                print(f'[WARN] Shape inference failed or onnx not available: {si_e}. Proceeding with original float ONNX.')
                quant_target = float_export_path

            from onnxruntime.quantization import quantize_dynamic, QuantType
            print('[INFO] Quantizing ONNX model with onnxruntime.quantization.quantize_dynamic')
            quantize_dynamic(quant_target, quant_export_path, weight_type=QuantType.QInt8)
            print(f'[INFO] Quantized ONNX saved to {quant_export_path}')
        except Exception as e:
            print(f"[WARN] ONNX quantization failed or onnxruntime.quantization not available: {e}")
            # If quantization fails, fall back to keeping the float ONNX as the export
            quant_export_path = float_export_path

    except Exception as e:
        print(f"[ERROR] Failed to export ONNX: {e}")
        raise

# Validate final ONNX model if onnx is installed
final_export = quant_export_path if os.path.exists(quant_export_path) else float_export_path
try:
    import onnx
    m = onnx.load(final_export)
    onnx.checker.check_model(m)
    print('ONNX model validated successfully:', final_export)
except Exception as e:
    print(f"[WARN] ONNX validation failed or onnx not installed: {e}")