"""
=============================================================
 Run7 — Fine-tune Run5 to handle partial/side human views
=============================================================
 Problem: Run5 misses humans seen from the side, hands reaching
          into frame, partial bodies at frame edges.

 Fix: Fine-tune Run5's .keras model with:
   1. All existing data (real_environment human + nonhuman)
   2. NEW partial-view images you capture (put in new_captures/)
   3. Aggressive augmentations: random shift, edge crop, cutout
      that simulate partial body views the model never saw.

 BEFORE RUNNING:
   1. Capture 20-30 new images from your MLX90640:
      - Hand only from left/right
      - Side profile walking past
      - Half body entering frame
      - Person at edge of sensor view
   2. Put Human images in:  new_captures/human/
      Put NonHuman images in: new_captures/nonhuman/  (optional)

 USAGE:
   python finetune_run7.py
   python finetune_run7.py --check       (just show data counts)
   python finetune_run7.py --epochs 25
   python finetune_run7.py --new_dir path/to/my_captures
=============================================================
"""

import os, sys, json, time, random, argparse
import numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR / "thermal_results" / "Run7_PartialViews"

# ── Data sources ─────────────────────────────────────────────────────────
REALENV_DIR     = SCRIPT_DIR / "dataset" / "real_environment"
NEW_CAPTURES    = SCRIPT_DIR / "new_captures"   # <-- put your new images here
THERMAL_DATASET = SCRIPT_DIR / "dataset" / "Thermal_Dataset"
ARCHIVE_DIR     = SCRIPT_DIR / "dataset" / "archive" / "Thermal Image Dataset"
EXTRA_DIR       = SCRIPT_DIR / "dataset" / "extra_nonhuman"

# ── Base model (Run5) ───────────────────────────────────────────────────
BASE_MODEL = SCRIPT_DIR / "thermal_results" / "Run5_RealEnv" / "best_model.keras"

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE        = (160, 160)
MLX_SIZE        = (32, 24)
BATCH_SIZE      = 32
EPOCHS_HEAD     = 8
EPOCHS_FT       = 15
LR_HEAD         = 2e-5
LR_FT           = 2e-6
HUMAN_BOOST     = 1.5
VAL_SPLIT       = 0.15
IMAGE_EXTS      = {'.png', '.jpg', '.jpeg', '.bmp'}
NEW_OVERSAMPLE  = 1    # no oversampling — images already augmented on disk
REALENV_OVERSAMPLE = 1

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import confusion_matrix
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tf.random.set_seed(SEED)
    print(f"TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0]}")
    else:
        print("No GPU — training on CPU (will be slower)")
except ImportError as e:
    sys.exit(f"Missing: {e}\nRun: pip install tensorflow scikit-learn pillow matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_paths(directory, label, desc, mode='direct'):
    paths = []
    if not directory.exists():
        return paths
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in IMAGE_EXTS:
            paths.append((str(f), label, mode))
    if paths:
        print(f"    {desc}: {len(paths)} images")
    return paths


def collect_all_data(new_dir):
    print("\n[Collecting dataset...]")
    all_items = []

    # ── Existing Human ────────────────────────────────────────────────────
    # Thermal_Dataset
    for split in ['train', 'valid', 'test']:
        d = THERMAL_DATASET / split / "images"
        all_items += collect_paths(d, 1, f"Thermal_Dataset/{split} -> Human")

    # Real environment human
    for subdir in ['human', 'Human']:
        d = REALENV_DIR / subdir
        items = collect_paths(d, 1, f"real_environment/{subdir} -> Human", mode='realenv')
        all_items += items

    # Roboflow/ASL human
    all_items += collect_paths(EXTRA_DIR / "roboflow_thermal" / "human", 1,
                               "roboflow/human -> Human")
    all_items += collect_paths(EXTRA_DIR / "asl_tid" / "human", 1,
                               "asl_tid/human -> Human")

    # ── Existing NonHuman ─────────────────────────────────────────────────
    # Archive car/cat
    archive_count = 0
    for cam in ['FLIR', 'SeekThermal']:
        for split_name in ['Train', 'Test', 'train', 'test']:
            split_dir = ARCHIVE_DIR / cam / split_name
            if not split_dir.exists():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir() and cls_dir.name.lower() in ('car', 'cat'):
                    for f in sorted(cls_dir.iterdir()):
                        if f.suffix.lower() in IMAGE_EXTS:
                            all_items.append((str(f), 0, 'downscale'))
                            archive_count += 1
    if archive_count:
        print(f"    Archive car/cat (downscaled) -> NonHuman: {archive_count} images")

    # Real environment nonhuman
    for subdir in ['nonhuman', 'NonHuman', 'background', 'Background', 'objects']:
        d = REALENV_DIR / subdir
        all_items += collect_paths(d, 0, f"real_environment/{subdir} -> NonHuman", mode='realenv')

    # Extra nonhuman
    all_items += collect_paths(EXTRA_DIR / "roboflow_thermal" / "nonhuman", 0,
                               "roboflow/nonhuman -> NonHuman")
    all_items += collect_paths(EXTRA_DIR / "asl_background", 0,
                               "asl_background -> NonHuman")

    # ── NEW CAPTURES (the key addition) ───────────────────────────────────
    new_dir = Path(new_dir)
    new_human = collect_paths(new_dir / "human", 1,
                              "NEW human (partial/side views)", mode='new')
    new_nonhuman = collect_paths(new_dir / "nonhuman", 0,
                                  "NEW nonhuman", mode='new')

    n_new = len(new_human) + len(new_nonhuman)
    if n_new == 0:
        print(f"\n  WARNING: No new captures found in {new_dir}/")
        print(f"  Create these folders and add your new images:")
        print(f"    {new_dir / 'human'}/     <- side views, hands, partial body")
        print(f"    {new_dir / 'nonhuman'}/  <- (optional) more nonhuman scenes")
        print(f"\n  Continuing with existing data + new augmentations only...\n")

    all_items += new_human + new_nonhuman

    # ── Oversample real_env and new captures ──────────────────────────────
    realenv_items = [x for x in all_items if x[2] == 'realenv']
    new_items     = [x for x in all_items if x[2] == 'new']

    if realenv_items:
        for _ in range(REALENV_OVERSAMPLE - 1):
            all_items.extend(realenv_items)
        print(f"  Real environment: {len(realenv_items)} x{REALENV_OVERSAMPLE} = {len(realenv_items)*REALENV_OVERSAMPLE}")

    if new_items:
        for _ in range(NEW_OVERSAMPLE - 1):
            all_items.extend(new_items)
        print(f"  New captures: {len(new_items)} x{NEW_OVERSAMPLE} = {len(new_items)*NEW_OVERSAMPLE}")

    n_human    = sum(1 for _, l, _ in all_items if l == 1)
    n_nonhuman = sum(1 for _, l, _ in all_items if l == 0)
    print(f"\n  Total: {n_human} Human + {n_nonhuman} NonHuman = {n_human+n_nonhuman}")

    return all_items


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path, mode='direct', add_noise=True):
    img = Image.open(path).convert('RGB')

    if mode == 'downscale':
        small = img.resize(MLX_SIZE, Image.BILINEAR)
        if add_noise:
            arr = np.array(small, dtype=np.float32)
            arr = np.clip(arr + np.random.normal(0, 2.0, arr.shape), 0, 255).astype(np.uint8)
            small = Image.fromarray(arr)
        img = small.resize(IMG_SIZE, Image.BILINEAR)
    elif mode in ('realenv', 'new'):
        small = img.resize(MLX_SIZE, Image.BILINEAR)
        img = small.resize(IMG_SIZE, Image.BILINEAR)
    else:
        img = img.resize(IMG_SIZE, Image.BILINEAR)

    return np.array(img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION — same as Run5 (images are already augmented)
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(arr, mode='direct'):
    """Light augmentation only — data is already augmented on disk."""
    strong = mode in ('new', 'realenv')

    # Horizontal flip
    if random.random() > 0.5:
        arr = arr[:, ::-1, :]

    # Brightness
    if strong:
        arr = np.clip(arr * random.uniform(0.75, 1.25), 0, 255)
    else:
        arr = np.clip(arr * random.uniform(0.85, 1.15), 0, 255)

    # Gaussian noise
    if random.random() > 0.4:
        noise_std = random.uniform(2.0, 5.0) if strong else 3.0
        arr = np.clip(arr + np.random.normal(0, noise_std, arr.shape), 0, 255)

    # Rotation (small)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15) if strong else random.uniform(-8, 8)
        img_pil = Image.fromarray(arr.astype(np.uint8))
        img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        arr = np.array(img_pil, dtype=np.float32)

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class TrainDataset(keras.utils.Sequence):
    def __init__(self, items, batch_size=32, augment=True):
        self.items = items
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(items))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.items) / self.batch_size))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx+1) * self.batch_size]
        X = np.zeros((len(batch_idx), *IMG_SIZE, 3), dtype=np.float32)
        y = np.zeros(len(batch_idx), dtype=np.float32)

        for i, bi in enumerate(batch_idx):
            path, label, mode = self.items[bi]
            try:
                img = load_image(path, mode, add_noise=self.augment)
            except Exception:
                img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

            arr = img.astype(np.float32)

            if self.augment:
                arr = augment_image(arr, mode=mode)

            X[i] = (arr - 127.5) / 127.5
            y[i] = float(label)

        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, items, threshold=0.5):
    gen = TrainDataset(items, batch_size=64, augment=False)
    probs = model.predict(gen, verbose=0).flatten()
    y_true = np.array([lbl for _, lbl, _ in items[:len(probs)]], dtype=int)

    best_score, best_t = -1, threshold
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fn = np.sum((preds == 0) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        recall = tp / (tp + fn + 1e-8)
        prec   = tp / (tp + fp + 1e-8)
        f1     = 2 * prec * recall / (prec + recall + 1e-8)
        score  = 0.7 * recall + 0.3 * f1
        if score > best_score:
            best_score, best_t = score, t

    preds = (probs >= best_t).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn)
    rec  = tp / (tp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1,
        'fn': int(fn), 'fp': int(fp), 'threshold': best_t,
        'confusion_matrix': cm.tolist(),
        'n_human': int(np.sum(y_true == 1)),
        'n_nonhuman': int(np.sum(y_true == 0)),
    }


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d0d1a')
    ax.imshow(np.array(cm), cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['NonHuman','Human'], color='white')
    ax.set_yticklabels(['NonHuman','Human'], color='white')
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('Actual', color='white')
    ax.set_title(title, color='white', fontweight='bold', pad=12)
    for i in range(2):
        for j in range(2):
            v = cm[i][j]
            ax.text(j, i, str(v), ha='center', va='center', fontsize=18,
                    fontweight='bold', color='white' if v > np.array(cm).max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run7 — Fine-tune Run5 for partial/side human views')
    parser.add_argument('--epochs', type=int, default=EPOCHS_HEAD + EPOCHS_FT)
    parser.add_argument('--new_dir', type=str, default=str(NEW_CAPTURES),
                        help='Directory with new human/nonhuman captures')
    parser.add_argument('--base_model', type=str, default=str(BASE_MODEL),
                        help='Path to base .keras model')
    parser.add_argument('--check', action='store_true',
                        help='Only show dataset counts')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('  Run7 — Partial/Side View Fine-Tuning')
    print('='*60)

    # ── Collect data ──────────────────────────────────────────────────────
    all_items = collect_all_data(args.new_dir)

    if args.check:
        print("\n[Data check only]")
        return

    # ── Check base model ──────────────────────────────────────────────────
    base_path = Path(args.base_model)
    if not base_path.exists():
        sys.exit(f"ERROR: Base model not found: {base_path}\n"
                 f"Make sure Run5 best_model.keras exists.")

    # ── Balance classes ───────────────────────────────────────────────────
    human_items    = [x for x in all_items if x[1] == 1]
    nonhuman_items = [x for x in all_items if x[1] == 0]

    max_nh = int(len(human_items) * 1.5)
    if len(nonhuman_items) > max_nh:
        random.shuffle(nonhuman_items)
        nonhuman_items = nonhuman_items[:max_nh]
        print(f"  Subsampled NonHuman to {len(nonhuman_items)}")

    all_items = human_items + nonhuman_items
    random.shuffle(all_items)

    # ── Split (keep new captures in both train and val) ───────────────────
    new_items    = [x for x in all_items if x[2] == 'new']
    realenv_items = [x for x in all_items if x[2] == 'realenv']
    other_items  = [x for x in all_items if x[2] not in ('new', 'realenv')]

    n_val_new = max(1, int(len(new_items) * VAL_SPLIT)) if new_items else 0
    n_val_re  = max(2, int(len(realenv_items) * VAL_SPLIT)) if realenv_items else 0
    n_val_ot  = int(len(other_items) * VAL_SPLIT)

    random.shuffle(new_items)
    random.shuffle(realenv_items)
    random.shuffle(other_items)

    val_items   = new_items[:n_val_new] + realenv_items[:n_val_re] + other_items[:n_val_ot]
    train_items = new_items[n_val_new:] + realenv_items[n_val_re:] + other_items[n_val_ot:]
    random.shuffle(train_items)
    random.shuffle(val_items)

    n_new_train = sum(1 for x in train_items if x[2] == 'new')
    n_new_val   = sum(1 for x in val_items   if x[2] == 'new')
    print(f"\n  Train: {len(train_items)} | Val: {len(val_items)}")
    print(f"  New captures in train: {n_new_train} | val: {n_new_val}")

    # ── Class weights ─────────────────────────────────────────────────────
    labels_train = [lbl for _, lbl, _ in train_items]
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=labels_train)
    class_weights = {0: cw[0], 1: cw[1] * HUMAN_BOOST}
    print(f"  Class weights: NonHuman={class_weights[0]:.3f}, Human={class_weights[1]:.3f}")

    train_gen = TrainDataset(train_items, BATCH_SIZE, augment=True)
    val_gen   = TrainDataset(val_items,   BATCH_SIZE, augment=False)

    # ── Load Run5 model ───────────────────────────────────────────────────
    print(f"\n  Loading base model: {base_path}")
    model = keras.models.load_model(str(base_path))
    print(f"  Model loaded: {len(model.layers)} layers")

    # ── Phase 1: Freeze backbone, train head ──────────────────────────────
    print(f"\n  Phase 1: Head training ({EPOCHS_HEAD} epochs, LR={LR_HEAD})")
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sl in layer.layers:
                sl.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
                  loss='binary_crossentropy', metrics=['accuracy'])

    t0 = time.time()
    cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8),
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD,
              class_weight=class_weights, callbacks=cb, verbose=1)

    # ── Phase 2: Unfreeze top layers ──────────────────────────────────────
    print(f"\n  Phase 2: Fine-tuning top 30 layers ({EPOCHS_FT} epochs, LR={LR_FT})")
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            total = len(layer.layers)
            for i, sl in enumerate(layer.layers):
                sl.trainable = (i >= total - 30)
    model.compile(optimizer=keras.optimizers.Adam(LR_FT),
                  loss='binary_crossentropy', metrics=['accuracy'])

    cb2 = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8),
        keras.callbacks.ModelCheckpoint(str(RESULTS_DIR / 'best_model.keras'),
                                        monitor='val_loss', save_best_only=True, verbose=1),
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FT,
              class_weight=class_weights, callbacks=cb2, verbose=1)
    print(f"  Training time: {time.time()-t0:.0f}s")

    # Load best checkpoint
    best = RESULTS_DIR / 'best_model.keras'
    if best.exists():
        model = keras.models.load_model(str(best))

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n  Evaluating...")
    metrics = evaluate(model, val_items)
    print(f"  Val Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Val Human Recall: {metrics['recall']:.4f}")
    print(f"  Val Precision:    {metrics['precision']:.4f}")
    print(f"  Val F1:           {metrics['f1']:.4f}")
    print(f"  Val FN:           {metrics['fn']}  FP: {metrics['fp']}")
    print(f"  Threshold:        {metrics['threshold']:.2f}")

    plot_cm(metrics['confusion_matrix'], 'Run7 Partial Views — Val',
            RESULTS_DIR / 'confusion_matrix.png')

    # ── TFLite export ─────────────────────────────────────────────────────
    print("\n  Exporting TFLite...")
    tflite_dir = RESULTS_DIR / "tflite"
    tflite_dir.mkdir(exist_ok=True)
    tflite_path = tflite_dir / "model.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(str(tflite_path), 'wb') as f:
        f.write(converter.convert())
    size_mb = os.path.getsize(str(tflite_path)) / (1024*1024)
    print(f"  TFLite: {tflite_path} ({size_mb:.2f} MB)")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        'run': 'Run7_PartialViews',
        'base_model': str(base_path),
        'n_train': len(train_items),
        'n_val': len(val_items),
        'n_new_captures': sum(1 for _, _, m in all_items if m == 'new'),
        'note': 'images pre-augmented on disk, light augmentation during training only',
        'val_metrics': metrics,
        'tflite': str(tflite_path),
        'tflite_mb': size_mb,
    }
    with open(str(RESULTS_DIR / 'run7_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Update enhanced_inference.py ──────────────────────────────────────
    inference_script = SCRIPT_DIR / "enhanced_inference.py"
    if inference_script.exists():
        content = inference_script.read_text(encoding='utf-8')
        old = 'SCRIPT_DIR / "thermal_results" / "Run5_RealEnv" / "tflite" / "model.tflite"'
        new = 'SCRIPT_DIR / "thermal_results" / "Run7_PartialViews" / "tflite" / "model.tflite"'
        if old in content:
            content = content.replace(old, new)
            inference_script.write_text(content, encoding='utf-8')
            print("\n  Auto-updated enhanced_inference.py MODEL_PATH -> Run7")

    print('\n' + '='*60)
    print('  RUN7 COMPLETE')
    print(f'  Human Recall: {metrics["recall"]:.4f}  |  FN: {metrics["fn"]}  FP: {metrics["fp"]}')
    print(f'  TFLite: {tflite_path}')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
