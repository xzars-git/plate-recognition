#!/usr/bin/env python
"""
🎨 Train Color Classifier
Train lightweight MobileNetV2 untuk classify warna plat nomor
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def create_color_classifier(input_shape=(96, 96, 3), num_classes=4):
    """
    Create lightweight color classifier based on MobileNetV2
    
    Args:
        input_shape: Input image shape
        num_classes: Number of color classes (4: white, black, red, yellow)
    
    Returns:
        Keras model
    """
    
    # Use MobileNetV2 as backbone (pre-trained on ImageNet)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=0.35  # Width multiplier (smaller = faster)
    )
    
    # Freeze base model layers (transfer learning)
    base_model.trainable = False
    
    # Build classifier
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing
    x = layers.Rescaling(1./255)(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def train_color_classifier(
    dataset_dir='dataset/plate_colors',
    output_dir='models/color_classifier',
    epochs=30,
    batch_size=32,
    input_size=96
):
    """Train color classifier"""
    
    print("="*70)
    print("🎨 TRAINING COLOR CLASSIFIER")
    print("="*70)
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check dataset
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    
    if not train_dir.exists():
        print(f"\n❌ ERROR: Training directory not found: {train_dir}")
        print("   Run: python prepare_color_dataset.py first!")
        return
    
    # Count samples
    color_names = ['white', 'black', 'red', 'yellow']
    train_counts = {c: len(list((train_dir / c).glob('*.jpg'))) for c in color_names}
    val_counts = {c: len(list((val_dir / c).glob('*.jpg'))) for c in color_names}
    
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Train: {total_train} images")
    for color in color_names:
        print(f"      ⚪ {color.capitalize()}: {train_counts[color]}")
    print(f"   Val: {total_val} images")
    for color in color_names:
        print(f"      ⚪ {color.capitalize()}: {val_counts[color]}")
    
    if total_train < 50:
        print(f"\n⚠️ WARNING: Only {total_train} training samples")
        print("   Recommended: 200+ samples for good accuracy")
        print("   Continue anyway? (Model may underfit)")
    
    # Data augmentation
    print(f"\n⚙️ Configuration:")
    print(f"   Input size: {input_size}x{input_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Classes: {len(color_names)}")
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=color_names,
        batch_size=batch_size,
        image_size=(input_size, input_size),
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=color_names,
        batch_size=batch_size,
        image_size=(input_size, input_size),
        shuffle=False,
        seed=42
    )
    
    # Data augmentation (only for training)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])
    
    # Apply augmentation
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Performance optimization
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Create model
    print(f"\n🤖 Building model...")
    model = create_color_classifier(
        input_shape=(input_size, input_size, 3),
        num_classes=len(color_names)
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_path / 'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            output_path / 'training_log.csv'
        )
    ]
    
    # Train
    print(f"\n🔥 Starting training...")
    print("="*70)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Evaluate
    print(f"\n📊 Final Evaluation:")
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    
    print(f"   Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Val Accuracy: {val_acc*100:.2f}%")
    
    # Save final model
    model.save(output_path / 'final_model.h5')
    print(f"\n💾 Model saved:")
    print(f"   Best: {output_path / 'best_model.h5'}")
    print(f"   Final: {output_path / 'final_model.h5'}")
    
    # Plot training history
    print(f"\n📈 Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_history.png', dpi=150)
    print(f"   Saved: {output_path / 'training_history.png'}")
    
    # Next steps
    print(f"\n🎯 Next Steps:")
    print(f"   1. Convert to TFLite: python convert_to_tflite.py")
    print(f"   2. Test model: python test_color_classifier.py")
    print(f"   3. Integrate to pipeline: python test_full_pipeline.py")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train color classifier')
    parser.add_argument('--dataset', default='dataset/plate_colors',
                       help='Dataset directory')
    parser.add_argument('--output', default='models/color_classifier',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--size', type=int, default=96,
                       help='Input image size')
    
    args = parser.parse_args()
    
    model, history = train_color_classifier(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=args.size
    )
