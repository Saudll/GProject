"""
=============================================================
 MLX90640 Thermal Image Tester
 Tests the trained MobileNetV2 TFLite model on a real
 MLX90640-D110 image (native 32×24 resolution).
=============================================================

USAGE:
    python test_mlx_image.py --image your_image.png
    python test_mlx_image.py --image thermal_frame.npy     # raw temperature array
    python test_mlx_image.py --image your_image.png --threshold 0.4
    python test_mlx_image.py --image your_image.png --show  # display result visually

SUPPORTED INPUT FORMATS:
    .png / .jpg / .jpeg   — Image already captured from MLX90640 (any pixel size)
    .npy                  — Raw numpy array of temperatures in °C (shape: 24×32 or 32×24)

The script handles the full preprocessing pipeline:
    MLX frame (32×24) → resize to 160×160 → normalise to [-1,1] → TFLite inference
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
MODEL_PATH   = SCRIPT_DIR / "thermal_results" / "Run5_RealEnv" / "tflite" / "model.tflite"
IMG_SIZE     = (160, 160)   # MobileNetV2 input
DEFAULT_THRESHOLD = 0.85    # Run4 optimal threshold (from threshold search on MLX-resolution data)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_mlx_npy(path: str) -> np.ndarray:
    """
    Load a raw MLX90640 temperature array (.npy file).
    Shape can be (24, 32) or (32, 24) or (768,) flat.
    Returns a uint8 RGB image (H, W, 3).
    """
    arr = np.load(path)

    # Handle flat array
    if arr.ndim == 1 and arr.size == 768:
        arr = arr.reshape(24, 32)

    # Handle transposed shape
    if arr.ndim == 2 and arr.shape == (32, 24):
        arr = arr.T  # → (24, 32)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected array shape: {arr.shape}. Expected (24,32) or (32,24).")

    # Normalise temperature range to 0–255
    t_min, t_max = arr.min(), arr.max()
    if t_max - t_min < 1e-6:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        norm = ((arr - t_min) / (t_max - t_min) * 255).astype(np.uint8)

    # Convert grayscale → RGB (stack 3 channels)
    rgb = np.stack([norm, norm, norm], axis=-1)  # (24, 32, 3)
    return rgb


def load_image(path: str) -> np.ndarray:
    """
    Load any image format (PNG, JPG, NPY).
    Returns uint8 RGB numpy array.
    """
    ext = Path(path).suffix.lower()

    if ext == '.npy':
        img_arr = load_mlx_npy(path)
        print(f"  Loaded .npy temperature array → shape {img_arr.shape}")
    else:
        pil_img = Image.open(path).convert('RGB')
        img_arr = np.array(pil_img)
        print(f"  Loaded image → original size: {pil_img.size[0]}×{pil_img.size[1]} px, mode: {pil_img.mode}")

    return img_arr


def preprocess(img_arr: np.ndarray) -> np.ndarray:
    """
    Full MobileNetV2 preprocessing pipeline.
    Input:  uint8 RGB array, any resolution
    Output: float32 array (1, 160, 160, 3), values in [-1, 1]
    """
    h, w = img_arr.shape[:2]

    # Step 1: Resize to 160×160 (BILINEAR = smoother upscaling from 32×24)
    pil = Image.fromarray(img_arr.astype(np.uint8)).resize(IMG_SIZE, Image.BILINEAR)
    resized = np.array(pil, dtype=np.float32)

    # Step 2: MobileNetV2 normalisation — scale [0,255] → [-1, 1]
    normalised = (resized - 127.5) / 127.5

    # Step 3: Add batch dimension → (1, 160, 160, 3)
    batch = np.expand_dims(normalised, axis=0)

    return batch, resized.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_tflite_model(model_path: str):
    """Load TFLite model. Tries tflite_runtime first, falls back to TensorFlow."""
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=str(model_path))
        print("  Using: tflite_runtime")
    except ImportError:
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            print("  Using: tensorflow.lite")
        except ImportError:
            sys.exit("ERROR: Install either 'tflite-runtime' or 'tensorflow' to run inference.")

    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, input_batch: np.ndarray) -> float:
    """Run inference and return Human probability (0–1)."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Cast to model's expected dtype
    expected_dtype = input_details[0]['dtype']
    input_data = input_batch.astype(expected_dtype)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output[0][0])  # sigmoid output → human probability
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualise(original_arr, resized_arr, prob, label, threshold, image_path):
    """Show the image, prediction confidence bar, and result."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    is_human  = label == 'Human'
    bar_color = '#00c853' if is_human else '#d50000'
    nh_prob   = 1 - prob

    # ── Left: original 32×24 ─────────────────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(original_arr, cmap='inferno', interpolation='nearest', aspect='auto')
    ax0.set_title(f'Original Input\n{original_arr.shape[1]}×{original_arr.shape[0]} px',
                  color='white', fontsize=11, fontweight='bold')
    ax0.axis('off')
    ax0.set_facecolor('#0d0d1a')

    # ── Middle: 160×160 preprocessed ─────────────────────────────────────
    ax1 = axes[1]
    ax1.imshow(resized_arr, cmap='inferno', aspect='auto')
    ax1.set_title(f'After Preprocessing\n160×160 px (model input)',
                  color='white', fontsize=11, fontweight='bold')
    ax1.axis('off')
    ax1.set_facecolor('#0d0d1a')

    # ── Right: prediction bar ─────────────────────────────────────────────
    ax2 = axes[2]
    ax2.set_facecolor('#0d0d1a')

    # Background bar
    ax2.barh(['NonHuman', 'Human'], [1.0, 1.0], color='#333355', height=0.5)

    # Actual probability bars
    ax2.barh(['NonHuman', 'Human'], [nh_prob, prob],
             color=['#ef5350', '#66bb6a'], height=0.5)

    # Threshold line
    ax2.axvline(x=threshold, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
    ax2.text(threshold + 0.02, 1.3, f'threshold={threshold}', color='yellow', fontsize=9)

    # Labels
    ax2.text(0.02, 1.0, f'{prob*100:.1f}%',    va='center', ha='left', fontsize=13, fontweight='bold', color='white')
    ax2.text(0.02, 0.0, f'{nh_prob*100:.1f}%', va='center', ha='left', fontsize=13, fontweight='bold', color='white')

    ax2.set_xlim(0, 1.2)
    ax2.set_xlabel('Probability', color='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444466')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#444466')
    for label_tick in ax2.get_xticklabels():
        label_tick.set_color('white')
    for label_tick in ax2.get_yticklabels():
        label_tick.set_color('white')

    ax2.set_title('Prediction', color='white', fontsize=11, fontweight='bold')

    # Big result label
    result_color = bar_color
    fig.text(0.5, 0.02,
             f'PREDICTION: {label.upper()}  (confidence: {max(prob, nh_prob)*100:.1f}%)',
             ha='center', va='bottom', fontsize=15, fontweight='bold',
             color=result_color, bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d0d1a', edgecolor=result_color))

    plt.suptitle(f'MLX90640 Thermal Inference  —  {Path(image_path).name}',
                 color='white', fontsize=13, y=1.01)
    plt.tight_layout()

    # Save result image
    out_path = Path(image_path).with_name(Path(image_path).stem + '_result.png')
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n  Result image saved: {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Test MobileNetV2 TFLite model on a real MLX90640 thermal image.'
    )
    parser.add_argument('--image',     required=True, help='Path to image (.png/.jpg) or temp array (.npy)')
    parser.add_argument('--model',     default=str(MODEL_PATH), help='Path to model.tflite')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Classification threshold (default {DEFAULT_THRESHOLD}). Lower = more sensitive to humans.')
    parser.add_argument('--show',      action='store_true', help='Display visualisation window')
    args = parser.parse_args()

    print('\n' + '═'*55)
    print('  MLX90640 Thermal Inference Test')
    print('═'*55)

    # Validate paths
    if not Path(args.image).exists():
        sys.exit(f"ERROR: Image not found: {args.image}")
    if not Path(args.model).exists():
        sys.exit(f"ERROR: Model not found: {args.model}\n"
                 f"Expected at: {MODEL_PATH}\n"
                 f"Run train_thermal.py first to generate the TFLite model.")

    # ── 1. Load image ─────────────────────────────────────────────────────
    print(f"\n[1/4] Loading image: {args.image}")
    original_arr = load_image(args.image)
    print(f"       Shape: {original_arr.shape}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print(f"\n[2/4] Preprocessing (resize to 160×160, normalise to [-1, 1])")
    input_batch, resized_arr = preprocess(original_arr)
    print(f"       Input tensor shape: {input_batch.shape}, dtype: {input_batch.dtype}")
    print(f"       Value range: [{input_batch.min():.3f}, {input_batch.max():.3f}]")

    # ── 3. Inference ──────────────────────────────────────────────────────
    print(f"\n[3/4] Running TFLite inference")
    print(f"       Model: {args.model}")
    interpreter = load_tflite_model(args.model)

    import time
    t0 = time.time()
    prob = predict(interpreter, input_batch)
    ms  = (time.time() - t0) * 1000

    label = 'Human' if prob >= args.threshold else 'NonHuman'

    # ── 4. Results ────────────────────────────────────────────────────────
    print(f"\n[4/4] Results")
    print(f"       Threshold:    {args.threshold}")
    print(f"       Human prob:   {prob:.4f}  ({prob*100:.1f}%)")
    print(f"       NonHuman prob:{1-prob:.4f}  ({(1-prob)*100:.1f}%)")
    print(f"       Inference:    {ms:.1f} ms")
    print()

    # Big clear result
    if label == 'Human':
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │   ✅  PREDICTION:  HUMAN DETECTED       │")
        print(f"  │       Confidence: {prob*100:.1f}%               │")
        print(f"  └─────────────────────────────────────────┘")
    else:
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │   ❌  PREDICTION:  NO HUMAN DETECTED    │")
        print(f"  │       Confidence: {(1-prob)*100:.1f}%               │")
        print(f"  └─────────────────────────────────────────┘")

    # Confidence note
    conf = max(prob, 1 - prob)
    if conf < 0.65:
        print(f"\n  ⚠  LOW CONFIDENCE ({conf*100:.1f}%). The model is uncertain.")
        print(f"     Try lowering --threshold (current: {args.threshold}) to improve human recall.")
    elif label == 'NonHuman' and prob > 0.30:
        print(f"\n  ⚠  Human probability is {prob*100:.1f}% — close to borderline.")
        print(f"     Consider lowering --threshold to {max(0.25, prob - 0.05):.2f} for rescue safety.")

    # ── 5. Visualise ──────────────────────────────────────────────────────
    if args.show:
        print(f"\n[+] Generating visualisation...")
        visualise(original_arr, resized_arr, prob, label, args.threshold, args.image)
    else:
        print(f"\n  Tip: Add --show to see a visual breakdown of the prediction.")

    print('\n' + '═'*55 + '\n')
    return prob, label


if __name__ == '__main__':
    main()
