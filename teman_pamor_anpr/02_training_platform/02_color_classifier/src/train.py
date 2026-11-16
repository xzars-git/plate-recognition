#!/usr/bin/env python
"""
Train Color Classifier
MobileNetV2 (alpha=0.35) untuk classify warna plat nomor.

Preprocessing (piksel 0-255 -> [-1, 1]) ada DI DALAM model, jadi TFLite
cukup diberi RGB float32 0-255 tanpa normalisasi manual di Flutter.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# red sengaja tidak default: baru ~8 sampel, tambahkan via --classes setelah dataset cukup
DEFAULT_CLASSES = ['white', 'black', 'yellow']


def create_color_classifier(input_shape=(96, 96, 3), num_classes=3):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=0.35,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1. / 127.5, offset=-1)(inputs)  # sesuai bobot ImageNet MobileNetV2
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def _count(split_dir, classes):
    return {c: len(list((split_dir / c).glob('*.jpg'))) for c in classes}


def _load(split_dir, classes, batch_size, size, shuffle):
    return tf.keras.preprocessing.image_dataset_from_directory(
        split_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=classes,
        batch_size=batch_size,
        image_size=(size, size),
        shuffle=shuffle,
        seed=42,
    )


def train_color_classifier(
    dataset_dir='dataset/plate_colors',
    output_dir='models/color_classifier',
    classes=DEFAULT_CLASSES,
    epochs=30,
    batch_size=32,
    input_size=96,
):
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_dir, val_dir = dataset_path / 'train', dataset_path / 'val'
    if not train_dir.exists():
        raise SystemExit(f"Training dir not found: {train_dir}. Run prepare_color_dataset.py first.")

    train_counts = _count(train_dir, classes)
    val_counts = _count(val_dir, classes)
    print(f"Train: {train_counts}\nVal:   {val_counts}")
    low = [c for c in classes if train_counts[c] < 50]
    if low:
        print(f"WARNING: kelas dengan <50 sampel train: {low}")

    train_ds = _load(train_dir, classes, batch_size, input_size, shuffle=True)
    val_ds = _load(val_dir, classes, batch_size, input_size, shuffle=False)

    # tanpa hue/saturation jitter: warna adalah label-nya
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ])
    def pad_aug(x):
        """Simulasi crop longgar (app memakai padding ~0.15): plat diperkecil lalu ditaruh di latar warna acak."""
        size = input_size
        new = tf.cast(tf.random.uniform([], 0.72, 1.0) * size, tf.int32)
        oy = tf.random.uniform([], 0, size - new + 1, tf.int32)
        ox = tf.random.uniform([], 0, size - new + 1, tf.int32)
        pads = [[0, 0], [oy, size - new - oy], [ox, size - new - ox], [0, 0]]
        small = tf.image.resize(x, (new, new))
        mask = tf.pad(tf.ones_like(small), pads)
        canvas = tf.random.uniform([tf.shape(x)[0], 1, 1, 3], 0, 255)
        out = tf.pad(small, pads) + (1.0 - mask) * canvas
        return tf.cond(tf.random.uniform([]) < 0.5, lambda: out, lambda: x)

    train_ds = train_ds.map(
        lambda x, y: (pad_aug(augment(x, training=True)), y), num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # dataset timpang (black ~65%): bobot kelas seimbang
    total = sum(train_counts.values())
    class_weight = {i: total / (len(classes) * max(train_counts[c], 1)) for i, c in enumerate(classes)}
    print(f"Class weight: {class_weight}")

    model = create_color_classifier((input_size, input_size, 3), len(classes))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    best_path = output_path / 'best_model.keras'
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.ModelCheckpoint(best_path, monitor='val_accuracy', save_best_only=True, mode='max'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            keras.callbacks.CSVLogger(output_path / 'training_log.csv'),
        ],
    )

    (output_path / 'labels.txt').write_text('\n'.join(classes) + '\n')

    # evaluasi per kelas dari model terbaik (akurasi total menipu di dataset timpang)
    best = keras.models.load_model(best_path)
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds])
    y_pred = np.argmax(best.predict(val_ds, verbose=0), axis=1)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    lines = [f"val accuracy: {(y_true == y_pred).mean() * 100:.2f}%", "rows=true, cols=pred", "\t" + "\t".join(classes)]
    for i, c in enumerate(classes):
        recall = cm[i, i] / max(cm[i].sum(), 1) * 100
        lines.append(f"{c}\t" + "\t".join(map(str, cm[i])) + f"\trecall={recall:.1f}%")
    report = '\n'.join(lines)
    (output_path / 'confusion_matrix.txt').write_text(report)
    print('\n' + report)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax, k, title in ((ax1, 'accuracy', 'Accuracy'), (ax2, 'loss', 'Loss')):
            ax.plot(history.history[k], label='train')
            ax.plot(history.history['val_' + k], label='val')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / 'training_history.png', dpi=120)
    except ImportError:
        pass

    print(f"\nSaved: {best_path}\nNext: python convert_to_tflite.py --model {best_path}")
    return best, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train color classifier')
    parser.add_argument('--dataset', default='dataset/plate_colors')
    parser.add_argument('--output', default='models/color_classifier')
    parser.add_argument('--classes', default=','.join(DEFAULT_CLASSES),
                        help='comma-separated, urutan = urutan output model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--size', type=int, default=96)
    args = parser.parse_args()

    train_color_classifier(
        dataset_dir=args.dataset,
        output_dir=args.output,
        classes=args.classes.split(','),
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=args.size,
    )
