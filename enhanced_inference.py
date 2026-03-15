"""
=============================================================
 Enhanced MLX90640 Inference — Two-Stage Hybrid Pipeline
=============================================================
 Stage 1: Physics-based thermal blob gate
   - Finds warm connected regions in the image
   - Checks if any blob matches human shape/size/position
   - If NO human-shaped warm blob exists → NonHuman immediately
   - Fridges, cold objects, blank scenes are rejected here

 Stage 2: MobileNetV2 TFLite neural network
   - Only runs if Stage 1 found a candidate blob
   - Deep shape/texture recognition
   - Both stages must agree to predict Human

USAGE:
    python enhanced_inference.py --image capture.png
    python enhanced_inference.py --image capture.npy    (raw temperature array)
    python enhanced_inference.py --image capture.png --show
    python enhanced_inference.py --image capture.png --threshold 0.75

SUPPORTED FORMATS:
    .png / .jpg   — Thermal image (any size, will be resized)
    .npy          — Raw MLX90640 temperature array (24×32 or 32×24, in Celsius)
=============================================================
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
MODEL_PATH  = SCRIPT_DIR / "thermal_results" / "Run5_RealEnv" / "tflite" / "model.tflite"
IMG_SIZE    = (160, 160)
MLX_SIZE    = (32, 24)       # native MLX90640 dimensions

# Neural network threshold (Run5 optimal)
NN_THRESHOLD = 0.35

# Blob gate parameters — tuned for 32×24 = 768 pixel frames
GATE_WARM_SIGMA      = 1.2   # warm pixels = mean + sigma * std
GATE_MIN_AREA_RATIO  = 0.020 # smallest blob = 2% of frame (≈15 pixels)
GATE_MAX_AREA_RATIO  = 0.70  # largest blob = 70% (human can fill most of frame)
GATE_MIN_ASPECT      = 0.55  # min height/width (humans taller than wide or square)
GATE_MIN_CENTROID_Y  = 0.20  # centroid must be at least 20% from top (not floating)
GATE_MIN_WARMTH      = 0.35  # blob pixels must be at least 35% of max intensity


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE CONNECTED COMPONENTS (no scipy needed — uses OpenCV if available)
# ─────────────────────────────────────────────────────────────────────────────
def connected_components(binary_mask):
    """
    Return (labeled_array, n_labels) for a 2D boolean mask.
    Uses OpenCV if available, otherwise pure-numpy BFS fallback.
    """
    try:
        import cv2
        mask_u8 = binary_mask.astype(np.uint8) * 255
        n, labeled = cv2.connectedComponents(mask_u8, connectivity=8)
        return labeled, n - 1   # subtract background label 0
    except ImportError:
        pass

    # Pure-numpy BFS fallback
    H, W = binary_mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    label = 0
    for r in range(H):
        for c in range(W):
            if binary_mask[r, c] and labeled[r, c] == 0:
                label += 1
                queue = [(r, c)]
                labeled[r, c] = label
                while queue:
                    y, x = queue.pop()
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if binary_mask[ny, nx] and labeled[ny, nx] == 0:
                                    labeled[ny, nx] = label
                                    queue.append((ny, nx))
    return labeled, label


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — THERMAL BLOB GATE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_thermal_blobs(img_array, verbose=False):
    """
    Analyze a thermal image for human-shaped warm blobs.

    Args:
        img_array: uint8 RGB (H, W, 3) — any size, will work on native 32x24
        verbose:   print detailed blob info

    Returns:
        dict with:
          'gate_pass'     : bool — True means a human-shaped warm blob was found
          'blobs'         : list of blob dicts with features
          'best_blob'     : best-scoring blob or None
          'warm_ratio'    : fraction of warm pixels in image
          'reason'        : human-readable explanation of gate decision
    """
    # Convert to grayscale (intensity = proxy for temperature)
    gray = np.mean(img_array.astype(np.float32), axis=2)
    H, W = gray.shape
    total_px = H * W

    mean_val = np.mean(gray)
    std_val  = np.std(gray)
    max_val  = np.max(gray)

    if std_val < 5.0:
        # Image is nearly uniform — no meaningful thermal gradient
        return {
            'gate_pass': False,
            'blobs': [],
            'best_blob': None,
            'warm_ratio': 0.0,
            'reason': f'Image too uniform (std={std_val:.1f}) — no thermal gradient detected'
        }

    # ── Find warm pixels ──────────────────────────────────────────────────
    threshold = mean_val + GATE_WARM_SIGMA * std_val
    warm_mask = gray > threshold
    warm_ratio = np.sum(warm_mask) / total_px

    if warm_ratio < 0.01:
        return {
            'gate_pass': False,
            'blobs': [],
            'best_blob': None,
            'warm_ratio': warm_ratio,
            'reason': f'No warm region found (warm pixels={warm_ratio*100:.1f}% < 1%)'
        }

    # ── Find connected warm blobs ─────────────────────────────────────────
    labeled, n_blobs = connected_components(warm_mask)

    blobs = []
    for blob_id in range(1, n_blobs + 1):
        blob = (labeled == blob_id)
        area = int(np.sum(blob))
        area_ratio = area / total_px

        # Skip tiny specks
        if area_ratio < GATE_MIN_AREA_RATIO * 0.5:
            continue

        rows_with_blob = np.any(blob, axis=1)
        cols_with_blob = np.any(blob, axis=0)

        if not np.any(rows_with_blob):
            continue

        rmin = int(np.argmax(rows_with_blob))
        rmax = int(H - 1 - np.argmax(rows_with_blob[::-1]))
        cmin = int(np.argmax(cols_with_blob))
        cmax = int(W - 1 - np.argmax(cols_with_blob[::-1]))

        blob_h = rmax - rmin + 1
        blob_w = cmax - cmin + 1
        aspect_ratio = blob_h / max(blob_w, 1)

        # Centroid (normalized)
        ys, xs = np.where(blob)
        centroid_y = float(np.mean(ys)) / H
        centroid_x = float(np.mean(xs)) / W

        # Warmth: mean intensity of blob vs max of image
        mean_blob_intensity = float(np.mean(gray[blob]))
        warmth = mean_blob_intensity / max(max_val, 1.0)

        # Compactness (circle = 1.0, elongated = lower)
        perimeter_approx = 2 * (blob_h + blob_w)
        compactness = (4 * np.pi * area) / max(perimeter_approx ** 2, 1)

        # ── Human criteria check ──────────────────────────────────────────
        fail_reasons = []
        if not (GATE_MIN_AREA_RATIO <= area_ratio <= GATE_MAX_AREA_RATIO):
            fail_reasons.append(f'area {area_ratio*100:.1f}% out of range [{GATE_MIN_AREA_RATIO*100:.0f}-{GATE_MAX_AREA_RATIO*100:.0f}%]')
        if aspect_ratio < GATE_MIN_ASPECT:
            fail_reasons.append(f'aspect_ratio {aspect_ratio:.2f} < {GATE_MIN_ASPECT} (too wide)')
        if centroid_y < GATE_MIN_CENTROID_Y:
            fail_reasons.append(f'centroid_y {centroid_y:.2f} < {GATE_MIN_CENTROID_Y} (too high in frame)')
        if warmth < GATE_MIN_WARMTH:
            fail_reasons.append(f'warmth {warmth:.2f} < {GATE_MIN_WARMTH} (not warm enough)')

        is_human_shaped = len(fail_reasons) == 0

        # Score (higher = more human-like)
        score = 0.0
        # Area score: peak at 15% of frame
        if GATE_MIN_AREA_RATIO <= area_ratio <= GATE_MAX_AREA_RATIO:
            score += 3.0 * (1 - abs(area_ratio - 0.15) / 0.15)
        # Aspect ratio score: peak at 2.0 (humans are tall)
        if aspect_ratio >= GATE_MIN_ASPECT:
            score += 2.0 * min(aspect_ratio / 2.0, 1.0)
        # Centroid score: middle of frame vertically
        if centroid_y >= GATE_MIN_CENTROID_Y:
            score += 1.5 * (1 - abs(centroid_y - 0.6) / 0.4)
        # Warmth score
        score += 2.0 * warmth

        blob_info = {
            'id': blob_id,
            'area': area,
            'area_ratio': area_ratio,
            'aspect_ratio': aspect_ratio,
            'centroid_y': centroid_y,
            'centroid_x': centroid_x,
            'warmth': warmth,
            'compactness': compactness,
            'is_human_shaped': is_human_shaped,
            'fail_reasons': fail_reasons,
            'score': score,
        }
        blobs.append(blob_info)

    if not blobs:
        return {
            'gate_pass': False,
            'blobs': [],
            'best_blob': None,
            'warm_ratio': warm_ratio,
            'reason': f'No blobs survived size filter (found {n_blobs} raw blobs)'
        }

    # Sort by score descending
    blobs.sort(key=lambda b: b['score'], reverse=True)
    best_blob = blobs[0]

    human_blobs = [b for b in blobs if b['is_human_shaped']]
    gate_pass = len(human_blobs) > 0

    if verbose:
        print(f"\n  [GATE] Warm pixels: {warm_ratio*100:.1f}%  |  Blobs found: {len(blobs)}")
        for i, b in enumerate(blobs[:3]):
            status = "✓ HUMAN" if b['is_human_shaped'] else "✗"
            print(f"  [GATE]  Blob {i+1}: area={b['area_ratio']*100:.1f}% "
                  f"aspect={b['aspect_ratio']:.2f} cy={b['centroid_y']:.2f} "
                  f"warmth={b['warmth']:.2f} score={b['score']:.1f}  {status}")
            if not b['is_human_shaped']:
                print(f"         Fail: {', '.join(b['fail_reasons'])}")

    if gate_pass:
        best_human = human_blobs[0]
        reason = (f"Human-shaped blob found: area={best_human['area_ratio']*100:.1f}%, "
                  f"aspect={best_human['aspect_ratio']:.2f}, "
                  f"cy={best_human['centroid_y']:.2f}, warmth={best_human['warmth']:.2f}")
    else:
        reason = f"No human-shaped blob. Best blob: {best_blob['fail_reasons']}"

    return {
        'gate_pass': gate_pass,
        'blobs': blobs,
        'best_blob': best_blob,
        'warm_ratio': warm_ratio,
        'reason': reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_mlx_npy(path):
    """Load raw MLX90640 .npy temperature array → uint8 RGB (32x24→160x160)."""
    arr = np.load(str(path))
    if arr.ndim == 1 and arr.size == 768:
        arr = arr.reshape(24, 32)
    if arr.ndim == 2 and arr.shape == (32, 24):
        arr = arr.T

    # Normalize temperatures to 0-255
    t_min, t_max = arr.min(), arr.max()
    if t_max - t_min < 0.5:
        t_min, t_max = t_min - 5, t_max + 5
    norm = ((arr - t_min) / (t_max - t_min) * 255).clip(0, 255).astype(np.uint8)

    # Stack to RGB
    rgb = np.stack([norm, norm, norm], axis=2)
    return rgb, arr   # rgb + raw temps


def load_image(path):
    """Load any thermal image as uint8 RGB array."""
    p = Path(path)
    if p.suffix.lower() == '.npy':
        rgb, raw_temps = load_mlx_npy(p)
        has_temps = True
    else:
        img = Image.open(str(p)).convert('RGB')
        rgb = np.array(img, dtype=np.uint8)
        raw_temps = None
        has_temps = False
    return rgb, raw_temps, has_temps


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — NEURAL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def load_tflite(model_path):
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=str(model_path))
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def nn_predict(interpreter, img_rgb, threshold):
    """
    Preprocess and run TFLite inference.
    Returns (probability_human, label, confidence_str)
    """
    # Per-channel normalization at native resolution: stretch each channel to 0-255
    # Makes NN robust to different thermal colormaps / capture settings
    src = img_rgb.astype(np.float32)
    for ch in range(src.shape[2]):
        lo, hi = src[:, :, ch].min(), src[:, :, ch].max()
        if hi - lo > 1.0:
            src[:, :, ch] = (src[:, :, ch] - lo) / (hi - lo) * 255.0
    src = np.clip(src, 0, 255).astype(np.uint8)

    # Resize to MLX size then to model input
    small = Image.fromarray(src).resize(MLX_SIZE, Image.BILINEAR)
    big   = small.resize(IMG_SIZE, Image.BILINEAR)
    arr   = np.array(big, dtype=np.float32)

    # MobileNetV2 normalization [-1, 1]
    arr = (arr - 127.5) / 127.5
    arr = np.expand_dims(arr, 0)

    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # Handle quantized model
    if inp_details[0]['dtype'] == np.int8:
        scale, zero_point = inp_details[0]['quantization']
        arr = (arr / scale + zero_point).clip(-128, 127).astype(np.int8)

    interpreter.set_tensor(inp_details[0]['index'], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(out_details[0]['index'])

    if output.dtype == np.int8:
        scale, zero_point = out_details[0]['quantization']
        prob = float(scale * (output.flatten()[0] - zero_point))
    else:
        prob = float(output.flatten()[0])

    prob = float(np.clip(prob, 0.0, 1.0))
    label = 'Human' if prob >= threshold else 'NonHuman'
    confidence = prob if prob >= threshold else (1.0 - prob)

    return prob, label, confidence


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: TEMPERATURE-BASED GATE (when .npy raw temps are available)
# ─────────────────────────────────────────────────────────────────────────────

def temperature_gate(raw_temps):
    """
    Uses actual temperature values (°C) from .npy file.
    Humans have surface temp ~33-36°C. Room ambient ~18-25°C.
    A human warm region should be >7°C above ambient.

    Returns: (pass, reason, stats)
    """
    flat = raw_temps.flatten()
    ambient = np.percentile(flat, 20)   # 20th percentile = ambient background
    max_temp = np.max(flat)
    temp_spread = max_temp - ambient

    # Fraction of pixels that are human-warm (>7°C above ambient)
    human_warm_threshold = ambient + 7.0
    warm_fraction = np.mean(flat > human_warm_threshold)

    stats = {
        'ambient_C': float(ambient),
        'max_C': float(max_temp),
        'spread_C': float(temp_spread),
        'warm_fraction': float(warm_fraction),
        'human_warm_threshold_C': float(human_warm_threshold),
    }

    if temp_spread < 4.0:
        return False, f'Temperature spread too small ({temp_spread:.1f}°C) — no heat source', stats

    if warm_fraction < 0.01:
        return False, f'No pixels above {human_warm_threshold:.1f}°C — no human-temperature region', stats

    if warm_fraction > 0.70:
        return False, f'{warm_fraction*100:.0f}% of pixels are warm — likely warm background, not human', stats

    return True, f'Warm region found: {warm_fraction*100:.1f}% pixels > {human_warm_threshold:.1f}°C (ambient={ambient:.1f}°C)', stats


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED DECISION
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_predict(image_path, nn_threshold=NN_THRESHOLD, verbose=False, show=False):
    """
    Full two-stage prediction pipeline.

    Returns:
        dict with full results including gate, nn, and final decision
    """
    print(f"\n{'─'*55}")
    print(f"  Image: {Path(image_path).name}")
    print(f"{'─'*55}")

    # ── Load image ────────────────────────────────────────────────────────
    t0 = time.time()
    img_rgb, raw_temps, has_temps = load_image(image_path)

    if verbose:
        print(f"  Loaded: {img_rgb.shape} dtype={img_rgb.dtype}")

    # ── Stage 1A: Temperature gate (only if .npy) ─────────────────────────
    temp_gate_pass = True
    temp_gate_reason = "N/A (PNG — no raw temperatures)"
    temp_stats = {}

    if has_temps:
        temp_gate_pass, temp_gate_reason, temp_stats = temperature_gate(raw_temps)
        icon = "✓" if temp_gate_pass else "✗"
        print(f"  [TEMP GATE] {icon} {temp_gate_reason}")
        if temp_stats:
            print(f"    Ambient={temp_stats['ambient_C']:.1f}°C  "
                  f"Max={temp_stats['max_C']:.1f}°C  "
                  f"WarmFraction={temp_stats['warm_fraction']*100:.1f}%")

    # ── Stage 1B: Blob shape gate ─────────────────────────────────────────
    # Resize to MLX native size for analysis (ensures consistent analysis scale)
    img_small = np.array(Image.fromarray(img_rgb).resize(MLX_SIZE, Image.BILINEAR))
    blob_result = analyze_thermal_blobs(img_small, verbose=verbose)

    blob_icon = "✓" if blob_result['gate_pass'] else "✗"
    print(f"  [BLOB GATE] {blob_icon} {blob_result['reason']}")

    # Combined Stage 1 result
    stage1_pass = temp_gate_pass and blob_result['gate_pass']

    if not stage1_pass:
        # Gate rejects — no need to run neural network
        reasons = []
        if not temp_gate_pass:
            reasons.append(f"Temp gate: {temp_gate_reason}")
        if not blob_result['gate_pass']:
            reasons.append(f"Blob gate: {blob_result['reason']}")

        final_label = "NonHuman"
        final_conf  = 0.95
        nn_prob     = None

        print(f"\n  ══ STAGE 1 REJECTED ══")
        print(f"  Prediction: {final_label}  (confidence: {final_conf*100:.0f}%)")
        print(f"  Neural network was NOT run (gate saved computation)")
        for r in reasons:
            print(f"    Reason: {r}")

    else:
        # ── Stage 2: Neural network ───────────────────────────────────────
        print(f"\n  [NN] Stage 1 passed → running neural network...")

        try:
            interpreter = load_tflite(MODEL_PATH)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print(f"  Check MODEL_PATH: {MODEL_PATH}")
            sys.exit(1)

        nn_prob, nn_label, nn_conf = nn_predict(interpreter, img_rgb, nn_threshold)
        nn_icon = "✓" if nn_label == "Human" else "✗"
        print(f"  [NN]  {nn_icon} Probability(Human) = {nn_prob*100:.1f}%  →  {nn_label}")

        # ── Final decision: BOTH must agree for Human ─────────────────────
        if nn_label == "Human":
            final_label = "Human"
            final_conf  = nn_conf
        else:
            final_label = "NonHuman"
            final_conf  = nn_conf

        print(f"\n  ══ FINAL DECISION ══")
        print(f"  Stage 1 (Gate): PASS")
        print(f"  Stage 2 (NN):   {nn_label} ({nn_prob*100:.1f}%)")
        print(f"  {'='*30}")
        print(f"  >> {final_label}  ({final_conf*100:.1f}% confidence)")

    elapsed_ms = (time.time() - t0) * 1000
    print(f"\n  Total time: {elapsed_ms:.1f} ms")

    result = {
        'image': str(image_path),
        'final_label': final_label,
        'final_confidence': final_conf,
        'stage1_pass': stage1_pass,
        'temp_gate_pass': temp_gate_pass,
        'temp_gate_reason': temp_gate_reason,
        'temp_stats': temp_stats,
        'blob_gate_pass': blob_result['gate_pass'],
        'blob_gate_reason': blob_result['reason'],
        'n_blobs': len(blob_result['blobs']),
        'best_blob': blob_result['best_blob'],
        'nn_prob': nn_prob,
        'elapsed_ms': elapsed_ms,
    }

    # ── Optional visualization ────────────────────────────────────────────
    if show:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor('#1a1a2e')

            # Original image
            img_disp = Image.fromarray(img_rgb)
            if img_disp.size != (IMG_SIZE[0], IMG_SIZE[1]):
                img_disp = img_disp.resize(IMG_SIZE, Image.BILINEAR)
            axes[0].imshow(img_disp, cmap='inferno')
            axes[0].set_title('Thermal Input (160x160)', color='white', fontsize=11)
            axes[0].axis('off')

            # Blob analysis on small image
            gray_small = np.mean(img_small.astype(np.float32), axis=2)
            axes[1].imshow(gray_small, cmap='inferno', vmin=gray_small.min(), vmax=gray_small.max())

            # Draw detected blobs
            if blob_result['blobs']:
                for b in blob_result['blobs'][:3]:
                    label_txt = f"A={b['area_ratio']*100:.0f}% AR={b['aspect_ratio']:.1f}"
                    color = 'lime' if b['is_human_shaped'] else 'red'
                    axes[1].text(
                        b['centroid_x'] * MLX_SIZE[0],
                        b['centroid_y'] * MLX_SIZE[1],
                        label_txt, color=color, fontsize=7, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
                    )

            axes[1].set_title(f'Blob Analysis (32x24 native)', color='white', fontsize=11)
            axes[1].axis('off')

            # Final verdict banner
            color = '#27AE60' if final_label == 'Human' else '#C0392B'
            fig.suptitle(
                f"{'🟢' if final_label=='Human' else '🔴'} {final_label}  "
                f"({final_conf*100:.1f}%)  —  Gate: {'PASS' if stage1_pass else 'FAIL'}",
                color=color, fontsize=16, fontweight='bold', y=0.02
            )

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  (matplotlib not available — skipping visualization)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BATCH TEST — run on a folder of images
# ─────────────────────────────────────────────────────────────────────────────

def batch_test(folder, nn_threshold=NN_THRESHOLD, verbose=False):
    """Test all images in a folder. Expects subfolders 'human' and 'nonhuman'."""
    folder = Path(folder)
    human_dir    = folder / 'human'
    nonhuman_dir = folder / 'nonhuman'

    results = {'human': [], 'nonhuman': []}

    for cls, d in [('human', human_dir), ('nonhuman', nonhuman_dir)]:
        if not d.exists():
            print(f"  No '{cls}' folder found at {d}")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.npy'):
                r = hybrid_predict(str(f), nn_threshold, verbose=verbose)
                results[cls].append(r)

    # Summary
    print(f"\n{'═'*55}")
    print("  BATCH TEST SUMMARY")
    print(f"{'═'*55}")

    for cls in ['human', 'nonhuman']:
        items = results[cls]
        if not items:
            continue
        correct = sum(1 for r in items if r['final_label'].lower() == cls)
        n = len(items)
        print(f"  {cls.upper():12s}: {correct}/{n} correct  ({correct/n*100:.1f}%)")
        for r in items:
            pred = r['final_label']
            ok = "✓" if pred.lower() == cls else "✗"
            gate = "G✓" if r['stage1_pass'] else "G✗"
            nn_str = f"NN={r['nn_prob']*100:.1f}%" if r['nn_prob'] is not None else "NN=skipped"
            print(f"    {ok} {gate}  {Path(r['image']).name:<25}  → {pred}  ({nn_str})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced MLX90640 Inference — Physics Gate + Neural Network'
    )
    parser.add_argument('--image',     type=str, help='Path to single image or .npy file')
    parser.add_argument('--folder',    type=str, help='Batch test folder (needs human/ and nonhuman/ subfolders)')
    parser.add_argument('--threshold', type=float, default=NN_THRESHOLD,
                        help=f'Neural network threshold (default: {NN_THRESHOLD})')
    parser.add_argument('--show',    action='store_true', help='Show visualization')
    parser.add_argument('--verbose', action='store_true', help='Print detailed blob analysis')
    args = parser.parse_args()

    if args.image:
        hybrid_predict(args.image, args.threshold, verbose=args.verbose, show=args.show)

    elif args.folder:
        batch_test(args.folder, args.threshold, verbose=args.verbose)

    else:
        parser.print_help()
        print("\n  Examples:")
        print("    python enhanced_inference.py --image capture.png --show")
        print("    python enhanced_inference.py --image temps.npy --verbose")
        print("    python enhanced_inference.py --folder test_images/ --verbose")


if __name__ == '__main__':
    main()
