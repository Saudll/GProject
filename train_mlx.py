"""
=============================================================
 MLX90640 Thermal Human Classifier — Train from Scratch
=============================================================
 Uses ONLY your MLX90640 captures. No FLIR, no SeekThermal.
 MobileNetV2 pretrained backbone + your real data.

 Data structure expected:
   dataset/
     human/        <- human thermal captures (.png/.jpg)
     nonhuman/     <- nonhuman thermal captures (.png/.jpg)

 INSTALL (Windows + RTX 2070 GPU):
   pip install tensorflow==2.10.1 scikit-learn pillow matplotlib

   Then install CUDA Toolkit 11.2 + cuDNN 8.1 from NVIDIA:
     CUDA: https://developer.nvidia.com/cuda-11.2.0-download-archive
     cuDNN: https://developer.nvidia.com/cudnn (needs free NVIDIA account)
     Copy cuDNN files into CUDA install folder.

   Or CPU-only (slower but no CUDA setup):
   pip install tensorflow scikit-learn pillow matplotlib

 USAGE:
   python train_mlx.py
   python train_mlx.py --data_dir dataset/real_environment
   python train_mlx.py --epochs 25 --batch_size 16
   python train_mlx.py --check    (just show dataset counts)
=============================================================
"""

import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

SCRIPT_DIR  = Path(__file__).parent
IMG_SIZE    = (160, 160)
MLX_SIZE    = (32, 24)       # native MLX90640 resolution
IMAGE_EXTS  = {'.png', '.jpg', '.jpeg', '.bmp'}

# Default paths
DEFAULT_DATA_DIR   = SCRIPT_DIR / "dataset" / "real_environment"
DEFAULT_RESULTS    = SCRIPT_DIR / "thermal_results" / "Run6_MLX_Only"


def parse_args():
    p = argparse.ArgumentParser(description="Train MLX90640 thermal classifier")
    p.add_argument('--data_dir',    type=str, default=str(DEFAULT_DATA_DIR),
                   help='Folder with human/ and nonhuman/ subfolders')
    p.add_argument('--results_dir', type=str, default=str(DEFAULT_RESULTS),
                   help='Where to save model and results')
    p.add_argument('--epochs',      type=int, default=20,
                   help='Total fine-tune epochs (default: 20)')
    p.add_argument('--batch_size',  type=int, default=16,
                   help='Batch size (default: 16)')
    p.add_argument('--lr',          type=float, default=1e-4,
                   help='Learning rate for fine-tuning (default: 1e-4)')
    p.add_argument('--model',       type=str, default='mobilenetv2',
                   choices=['mobilenetv2', 'efficientnetb0', 'resnet50',
                            'densenet121', 'nasnetmobile'],
                   help='Backbone model (default: mobilenetv2)')
    p.add_argument('--compare_all', action='store_true',
                   help='Train all 5 models and compare results')
    p.add_argument('--check',       action='store_true',
                   help='Just show dataset counts and exit')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_images(data_dir: Path):
    """Scan data_dir/human/ and data_dir/nonhuman/ for images."""
    human_dirs    = ['human', 'Human', 'person', 'Person']
    nonhuman_dirs = ['nonhuman', 'NonHuman', 'background', 'Background', 'objects']

    human_set = set()
    nonhuman_set = set()

    for name in human_dirs:
        d = data_dir / name
        if d.exists():
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith('._'):
                    human_set.add(str(f.resolve()))

    for name in nonhuman_dirs:
        d = data_dir / name
        if d.exists():
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith('._'):
                    nonhuman_set.add(str(f.resolve()))

    return sorted(human_set), sorted(nonhuman_set)


def show_dataset_info(human_paths, nonhuman_paths):
    """Print dataset summary."""
    n_h = len(human_paths)
    n_nh = len(nonhuman_paths)
    total = n_h + n_nh

    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Human:    {n_h} images")
    print(f"  NonHuman: {n_nh} images")
    print(f"  Total:    {total} images")

    if total > 0:
        print(f"  Balance:  {n_h/total*100:.0f}% Human / {n_nh/total*100:.0f}% NonHuman")

    if n_h < 20:
        print(f"\n  WARNING: Only {n_h} Human images. Aim for 50+ per class.")
    if n_nh < 20:
        print(f"\n  WARNING: Only {n_nh} NonHuman images. Aim for 50+ per class.")

    return n_h, n_nh


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(path):
    """Load image, ensure MLX resolution, resize to model input."""
    from PIL import Image

    img = Image.open(path).convert('RGB')

    # If not already MLX size, downscale to simulate MLX capture
    w, h = img.size
    if w != MLX_SIZE[0] or h != MLX_SIZE[1]:
        img = img.resize(MLX_SIZE, Image.BILINEAR)

    # Upscale to model input size
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ThermalDataset:
    """Data generator — no augmentation (dataset is pre-augmented)."""

    def __init__(self, paths, labels, batch_size, shuffle=True):
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(paths))

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.zeros((len(batch_idx), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        y = np.zeros(len(batch_idx), dtype=np.float32)

        for i, j in enumerate(batch_idx):
            arr = load_and_preprocess(self.paths[j])

            # MobileNetV2 normalization [-1, 1]
            X[i] = (arr - 127.5) / 127.5
            y[i] = float(self.labels[j])

        return X, y

    def to_tf_dataset(self):
        """Convert to tf.data.Dataset for GPU pipeline."""
        import tensorflow as tf

        def gen():
            self.on_epoch_end()
            for idx in range(len(self)):
                X, y = self[idx]
                yield X, y

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            )
        )
        return ds.prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    'mobilenetv2':   {'fn': 'MobileNetV2',      'size': '~3.4 MB TFLite', 'params': '3.4M'},
    'efficientnetb0':{'fn': 'EfficientNetB0',    'size': '~5.3 MB TFLite', 'params': '5.3M'},
    'resnet50':      {'fn': 'ResNet50',           'size': '~25 MB TFLite',  'params': '25.6M'},
    'densenet121':   {'fn': 'DenseNet121',        'size': '~8 MB TFLite',   'params': '8.1M'},
    'nasnetmobile':  {'fn': 'NASNetMobile',       'size': '~5.2 MB TFLite', 'params': '5.3M'},
}


def build_model(model_name='mobilenetv2'):
    """Build backbone + custom head for binary classification."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    apps = tf.keras.applications
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    if model_name == 'mobilenetv2':
        base = apps.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_name == 'efficientnetb0':
        base = apps.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_name == 'resnet50':
        base = apps.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_name == 'densenet121':
        base = apps.DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_name == 'nasnetmobile':
        base = apps.NASNetMobile(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        sys.exit(f"Unknown model: {model_name}")

    # Freeze entire backbone initially
    base.trainable = False

    # Custom classification head
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=x)
    return model, base


def unfreeze_top_layers(base, n_layers=30):
    """Unfreeze top N layers for fine-tuning."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"  Unfroze top {n_layers} layers ({trainable} trainable)")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import confusion_matrix, classification_report
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tf.random.set_seed(SEED)

    # ── GPU check ─────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n  GPU detected: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n  No GPU detected — training on CPU (slower)")

    # ── Collect data ──────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    human_paths, nonhuman_paths = collect_images(data_dir)
    n_h, n_nh = show_dataset_info(human_paths, nonhuman_paths)

    if n_h < 5 or n_nh < 5:
        sys.exit("ERROR: Need at least 5 images per class!")

    # Build arrays
    all_paths  = human_paths + nonhuman_paths
    all_labels = [1] * n_h + [0] * n_nh

    # ── Group-aware Train/Val split (80/20) ───────────────────────────────
    # Group augmentations by their base ID so all copies of one original
    # stay together (prevents data leakage between train and val)
    import re as _re

    def get_group_id(path):
        """Extract base group from filename: thermal_1773545574_orig.png -> thermal_1773545574"""
        name = Path(path).stem  # e.g. thermal_1773545574_aug_3
        # Strip _orig or _aug_N suffix
        name = _re.sub(r'_(orig|aug_\d+)$', '', name)
        return name

    # Build groups: {group_id: [(path, label), ...]}
    groups = {}
    for p, l in zip(all_paths, all_labels):
        gid = get_group_id(p)
        if gid not in groups:
            groups[gid] = {'label': l, 'paths': []}
        groups[gid]['paths'].append(p)

    # Split at the group level
    group_ids = sorted(groups.keys())
    group_labels = [groups[g]['label'] for g in group_ids]

    train_gids, val_gids = train_test_split(
        group_ids,
        test_size=0.2,
        stratify=group_labels,
        random_state=SEED,
    )

    # Flatten groups back to paths/labels
    train_paths, train_labels = [], []
    for gid in train_gids:
        for p in groups[gid]['paths']:
            train_paths.append(p)
            train_labels.append(groups[gid]['label'])

    val_paths, val_labels = [], []
    for gid in val_gids:
        for p in groups[gid]['paths']:
            val_paths.append(p)
            val_labels.append(groups[gid]['label'])

    n_groups = len(groups)
    print(f"\n  Unique originals: {n_groups} ({len(train_gids)} train / {len(val_gids)} val)")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")
    train_h = sum(train_labels)
    train_nh = len(train_labels) - train_h
    print(f"  Train split: {train_h} Human + {train_nh} NonHuman")

    # ── Class weights (handle imbalance) ──────────────────────────────────
    weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                   y=np.array(train_labels))
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"  Class weights: NonHuman={weights[0]:.2f}, Human={weights[1]:.2f}")

    # ── Data generators ───────────────────────────────────────────────────
    train_ds = ThermalDataset(train_paths, train_labels, args.batch_size, shuffle=True)
    val_ds   = ThermalDataset(val_paths, val_labels, args.batch_size, shuffle=False)

    train_tf = train_ds.to_tf_dataset()
    val_tf   = val_ds.to_tf_dataset()

    # ── Build model ───────────────────────────────────────────────────────
    model_name = args.model
    results_dir = Path(args.results_dir)
    if args.compare_all or model_name != 'mobilenetv2':
        results_dir = results_dir.parent / f"{results_dir.name}_{model_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    info = MODELS[model_name]
    model, base = build_model(model_name)
    print(f"\n  Model: {model_name} ({info['params']} params, {info['size']})")
    print(f"  Backbone frozen for Phase 1")

    # ── Phase 1: Train head only (5 epochs) ───────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Phase 1: Training head (backbone frozen)")
    print(f"{'='*50}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(results_dir / "best_model.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1,
    )

    head_epochs = min(5, args.epochs)
    model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=head_epochs,
        class_weight=class_weight,
        callbacks=[checkpoint],
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────
    ft_epochs = args.epochs - head_epochs
    if ft_epochs > 0:
        print(f"\n{'='*50}")
        print(f"  Phase 2: Fine-tuning top 30 backbone layers")
        print(f"{'='*50}")

        unfreeze_top_layers(base, n_layers=30)

        # Use very low LR for fine-tuning to preserve head calibration
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr * 0.1),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True,
        )
        lr_reduce = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        )

        # Recreate generators (reset epoch state)
        train_ds2 = ThermalDataset(train_paths, train_labels, args.batch_size, shuffle=True)
        val_ds2   = ThermalDataset(val_paths, val_labels, args.batch_size, shuffle=False)

        model.fit(
            train_ds2.to_tf_dataset(),
            validation_data=val_ds2.to_tf_dataset(),
            epochs=ft_epochs,
            class_weight=class_weight,
            callbacks=[checkpoint, early_stop, lr_reduce],
        )

    # ── Load best model ───────────────────────────────────────────────────
    best_path = results_dir / "best_model.keras"
    if best_path.exists():
        model = keras.models.load_model(str(best_path))
        print(f"\n  Loaded best model from {best_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Evaluation")
    print(f"{'='*50}")

    val_ds_eval = ThermalDataset(val_paths, val_labels, args.batch_size, shuffle=False)
    y_true = []
    y_prob = []

    for idx in range(len(val_ds_eval)):
        X, y = val_ds_eval[idx]
        preds = model.predict(X, verbose=0).flatten()
        y_true.extend(y.tolist())
        y_prob.extend(preds.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Show probability distribution so we can see model calibration
    human_probs = y_prob[y_true == 1]
    nonhuman_probs = y_prob[y_true == 0]
    print(f"\n  Probability distribution:")
    print(f"    Human images:    min={human_probs.min():.3f}  mean={human_probs.mean():.3f}  max={human_probs.max():.3f}")
    print(f"    NonHuman images: min={nonhuman_probs.min():.3f}  mean={nonhuman_probs.mean():.3f}  max={nonhuman_probs.max():.3f}")

    # Find optimal threshold
    best_thresh = 0.5
    best_f1 = 0
    for t in np.arange(0.05, 0.95, 0.05):
        y_pred_t = (y_prob >= t).astype(int)
        tp = np.sum((y_pred_t == 1) & (y_true == 1))
        fp = np.sum((y_pred_t == 1) & (y_true == 0))
        fn = np.sum((y_pred_t == 0) & (y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    y_pred = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp_count, fn_count, tp_count = cm.ravel()

    accuracy = (tp_count + tn) / len(y_true)
    recall   = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0

    print(f"\n  Optimal threshold: {best_thresh:.2f}")
    print(f"  Accuracy:  {accuracy*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}% (Human detection rate)")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  F1 Score:  {best_f1*100:.1f}%")
    print(f"  False Negatives: {fn_count} (humans missed)")
    print(f"  False Positives: {fp_count} (objects called human)")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred NH  Pred H")
    print(f"  Actual NH:   {tn:4d}    {fp_count:4d}")
    print(f"  Actual H:    {fn_count:4d}    {tp_count:4d}")

    # ── Save confusion matrix plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['NonHuman', 'Human'])
    ax.set_yticklabels(['NonHuman', 'Human'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (threshold={best_thresh:.2f})')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=18, color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(str(results_dir / "confusion_matrix.png"), dpi=100)
    print(f"\n  Saved: {results_dir / 'confusion_matrix.png'}")

    # ── Export TFLite ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Exporting TFLite")
    print(f"{'='*50}")

    tflite_dir = results_dir / "tflite"
    tflite_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = tflite_dir / "model.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(str(tflite_path), 'wb') as f:
        f.write(tflite_model)

    tflite_mb = os.path.getsize(str(tflite_path)) / (1024 * 1024)
    print(f"  TFLite model: {tflite_path}")
    print(f"  Size: {tflite_mb:.2f} MB")

    # ── Save results JSON ─────────────────────────────────────────────────
    results = {
        'run': results_dir.name,
        'model': model_name,
        'data_dir': str(data_dir),
        'n_human': n_h,
        'n_nonhuman': n_nh,
        'n_train': len(train_paths),
        'n_val': len(val_paths),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'optimal_threshold': float(best_thresh),
        'accuracy': float(accuracy),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(best_f1),
        'false_negatives': int(fn_count),
        'false_positives': int(fp_count),
        'confusion_matrix': cm.tolist(),
        'tflite_path': str(tflite_path),
        'tflite_mb': tflite_mb,
        'gpu': gpus[0].name if gpus else 'CPU',
    }

    results_json = results_dir / "results.json"
    with open(str(results_json), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Update enhanced_inference.py with new model path ──────────────────
    inference_script = SCRIPT_DIR / "enhanced_inference.py"
    if inference_script.exists():
        try:
            code = inference_script.read_text(encoding='utf-8')
            # Update model path
            import re
            code = re.sub(
                r'MODEL_PATH\s*=.*',
                f'MODEL_PATH  = SCRIPT_DIR / "thermal_results" / "{results_dir.name}" / "tflite" / "model.tflite"',
                code
            )
            # Update threshold
            code = re.sub(
                r'NN_THRESHOLD\s*=\s*[\d.]+',
                f'NN_THRESHOLD = {best_thresh}',
                code
            )
            inference_script.write_text(code, encoding='utf-8')
            print(f"\n  Updated enhanced_inference.py:")
            print(f"    MODEL_PATH -> {results_dir.name}/tflite/model.tflite")
            print(f"    NN_THRESHOLD -> {best_thresh}")
        except Exception as e:
            print(f"\n  Could not auto-update enhanced_inference.py: {e}")
            print(f"  Manually set MODEL_PATH and NN_THRESHOLD = {best_thresh}")

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"  Model:     {results_dir / 'best_model.keras'}")
    print(f"  TFLite:    {tflite_path} ({tflite_mb:.2f} MB)")
    print(f"  Threshold: {best_thresh}")
    print(f"  Accuracy:  {accuracy*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1:        {best_f1*100:.1f}%")
    print(f"\n  Copy tflite/model.tflite to your Raspberry Pi for deployment.")
    print(f"{'='*50}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    data_dir = Path(args.data_dir)
    human_paths, nonhuman_paths = collect_images(data_dir)

    if args.check:
        show_dataset_info(human_paths, nonhuman_paths)
        sys.exit(0)

    if len(human_paths) == 0 and len(nonhuman_paths) == 0:
        print(f"\n  No images found in {data_dir}")
        print(f"  Expected structure:")
        print(f"    {data_dir}/human/     <- human thermal captures")
        print(f"    {data_dir}/nonhuman/  <- nonhuman thermal captures")
        sys.exit(1)

    if args.compare_all:
        # Train all models and print comparison table
        all_results = []
        for name in MODELS:
            print(f"\n{'#'*60}")
            print(f"  TRAINING: {name} ({MODELS[name]['params']} params)")
            print(f"{'#'*60}")
            args.model = name
            result = train(args)
            all_results.append(result)

        # Print comparison table
        print(f"\n\n{'='*70}")
        print(f"  MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Model':<18} {'Accuracy':>8} {'Recall':>8} {'F1':>8} {'TFLite':>10} {'Threshold':>10}")
        print(f"  {'-'*62}")
        best_f1 = 0
        best_name = ''
        for r in all_results:
            tag = ''
            if r['f1'] > best_f1:
                best_f1 = r['f1']
                best_name = r['model']
            print(f"  {r['model']:<18} {r['accuracy']*100:>7.1f}% {r['recall']*100:>7.1f}% "
                  f"{r['f1']*100:>7.1f}% {r['tflite_mb']:>8.2f}MB {r['optimal_threshold']:>10.2f}")
        print(f"\n  BEST MODEL: {best_name} (F1={best_f1*100:.1f}%)")
        print(f"{'='*70}")

        # Save comparison
        comp_path = Path(args.results_dir).parent / "model_comparison.json"
        with open(str(comp_path), 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved: {comp_path}")
    else:
        train(args)
