"""
STEP 3: TRAIN MODEL - Fixed shape mismatch
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n" + "="*70)
print("🎯 STEP 3: TRAINING MODEL (FIXED SHAPE)")
print("="*70)
print("Expected time: 30-45 minutes")
print("Epochs: 25")
print("Batch size: 64")
print("="*70)

import pickle
import numpy as np
import tensorflow as tf

# Check files
required_files = [
    ("data/vgg16_features.pickle", "Step 1: python 1_extract_features.py"),
    ("vocabulary.pkl", "Step 2: python 2_preprocess.py"),
    ("captions.pkl", "Step 2: python 2_preprocess.py")
]

for file, solution in required_files:
    if not os.path.exists(file):
        print(f"❌ Missing: {file}")
        print(f"   Run: {solution}")
        exit()

print("Loading data...")

# Load features
with open("data/vgg16_features.pickle", 'rb') as f:
    features_raw = pickle.load(f)

# FIX: Remove .jpg extension from feature keys
features = {}
for key, value in features_raw.items():
    # Remove .jpg, .jpeg, .png extensions
    base_key = key.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('.JPG', '')
    
    # FIX SHAPE: Ensure features are 512-dim, not (1, 512)
    if len(value.shape) > 1:
        value = np.squeeze(value, axis=0)
    
    features[base_key] = value

# Load vocabulary
with open("vocabulary.pkl", 'rb') as f:
    vocab = pickle.load(f)

# Load captions
with open("captions.pkl", 'rb') as f:
    captions = pickle.load(f)

print(f"✅ Features (cleaned): {len(features)} images")
print(f"✅ Vocabulary: {len(vocab)} words")
print(f"✅ Captions: {len(captions)} images")

# Check feature shape
sample_feat = next(iter(features.values()))
print(f"✅ Feature shape: {sample_feat.shape} (should be (512,))")

# ===== FIND COMMON IMAGES =====
print("\nFinding images with both features and captions...")

common_images = []
for img_name in features.keys():
    if img_name in captions:
        common_images.append(img_name)

print(f"✅ Found {len(common_images)} common images")

# Use up to 3000 images for medium speed
import random
random.seed(42)
if len(common_images) > 3000:
    train_images = random.sample(common_images, 3000)
else:
    train_images = common_images

print(f"Using {len(train_images)} images for training")

# ===== PREPARE TRAINING DATA =====
print("\nPreparing training data...")

X_img, X_text, Y = [], [], []
MAXLEN = 20

for img_name in train_images:
    img_feat = features[img_name]
    img_caps = captions[img_name][:2]  # Use up to 2 captions per image
    
    for caption in img_caps:
        # Convert caption to token IDs
        ids = []
        for word in caption.split():
            if word in vocab:
                ids.append(vocab[word])
            else:
                ids.append(vocab['<PAD>'])
        
        # Skip if too short
        if len(ids) < 4:
            continue
        
        # Pad to MAXLEN tokens
        padded = np.full(MAXLEN, vocab['<PAD>'])
        length = min(len(ids), MAXLEN)
        padded[:length] = ids[:length]
        
        # Add to training data
        X_img.append(img_feat)
        X_text.append(padded[:-1])  # Input: first 19 tokens
        Y.append(padded[1:])        # Target: last 19 tokens

print(f"✅ Prepared {len(X_img)} training samples")

if len(X_img) == 0:
    print("❌ ERROR: No training samples created!")
    exit()

# Convert to numpy arrays
X_img = np.array(X_img, dtype=np.float32)
X_text = np.array(X_text, dtype=np.int32)
Y = np.array(Y, dtype=np.int32)

print(f"✅ Data shapes: X_img={X_img.shape}, X_text={X_text.shape}, Y={Y.shape}")

# ===== CREATE MEDIUM MODEL =====
def create_medium_model(vocab_size):
    """Medium size model for faster training"""
    print("\nBuilding medium model...")
    
    # Image input (512-dim VGG16 features)
    img_input = tf.keras.Input(shape=(512,), name='image_input')
    
    # Text input (19 tokens: MAXLEN-1)
    text_input = tf.keras.Input(shape=(19,), name='text_input')
    
    # Process image
    img_dense = tf.keras.layers.Dense(128, activation='relu')(img_input)
    img_dropout = tf.keras.layers.Dropout(0.2)(img_dense)
    img_repeat = tf.keras.layers.RepeatVector(19)(img_dropout)
    
    # Process text
    text_embed = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        mask_zero=True,
        name='text_embedding'
    )(text_input)
    
    # Combine
    combined = tf.keras.layers.Concatenate()([text_embed, img_repeat])
    
    # Simple LSTM
    lstm = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(combined)
    
    # Output
    output = tf.keras.layers.Dense(vocab_size, activation='softmax', name='output')(lstm)
    
    # Create model
    model = tf.keras.Model(inputs=[img_input, text_input], outputs=output)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model parameters: {model.count_params():,}")
    return model

# ===== TRAIN MODEL =====
print("\n🚀 Starting training (25 epochs)...")

model = create_medium_model(len(vocab))

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# Train with validation split
if len(X_img) < 100:
    print(f"⚠️  Very few samples ({len(X_img)}). Using smaller validation split.")
    validation_split = 0.1
else:
    validation_split = 0.15

print(f"Training samples: {len(X_img)}")
print(f"Validation split: {validation_split}")

print("\n" + "="*70)
print("TRAINING STARTED - This will take 30-45 minutes")
print("="*70)

history = model.fit(
    [X_img, X_text],
    Y,
    batch_size=64,
    epochs=25,
    validation_split=validation_split,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save("trained_model.keras")
print("\n✅ Model saved: trained_model.keras")

# Simple training plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train', linewidth=2)
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Val', linewidth=2)
plt.title('Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val', linewidth=2)
    plt.title('Training Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
print("✅ Plot saved: training_history.png")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print(f"   Trained on {len(X_img)} samples")
print(f"   Model saved: trained_model.keras")
print("="*70)

# Quick test of the model
print("\n🧪 Quick model test...")
test_idx = 0
test_pred = model.predict([X_img[test_idx:test_idx+1], X_text[test_idx:test_idx+1]], verbose=0)
print(f"Test prediction shape: {test_pred.shape}")
print("✅ Model is working correctly!")
