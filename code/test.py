"""
STEP 4: TEST MODEL - Simple testing
"""
import os
import pickle
import numpy as np
import tensorflow as tf

print("\n" + "="*70)
print("🎯 STEP 4: TESTING MODEL")
print("="*70)

# Check model exists
if not os.path.exists("trained_model.keras"):
    print("❌ No trained model found!")
    print("Run: python 3_train.py")
    exit()

print("Loading model...")
model = tf.keras.models.load_model("trained_model.keras", compile=False)

print("Loading vocabulary...")
with open("vocabulary.pkl", 'rb') as f:
    vocab = pickle.load(f)
reverse_vocab = {v: k for k, v in vocab.items()}

print("Loading features...")
with open("data/vgg16_features.pickle", 'rb') as f:
    features = pickle.load(f)

def generate_caption(image_name):
    """Generate caption for one image"""
    if image_name not in features:
        return "Image not found"
    
    feat = features[image_name]
    
    # FIX: Properly handle feature shape
    # Convert to numpy array first
    if isinstance(feat, tf.Tensor):
        feat = feat.numpy()
    
    # Ensure it's 2D: (1, 512)
    if len(feat.shape) == 1:
        feat = feat.reshape(1, -1)  # Now this works on numpy array
    elif len(feat.shape) == 2 and feat.shape[0] != 1:
        # If it's (n, 512), take mean or first
        feat = feat[0:1, :]
    
    # Start with <START>
    tokens = [vocab['<START>']]
    
    for _ in range(15):  # Max 15 words
        # Prepare input
        input_seq = np.full((1, 19), vocab['<PAD>'])
        length = min(len(tokens), 19)
        input_seq[0, :length] = tokens[:length]
        
        # FIX: Use proper tensor operations
        # Ensure feat is numpy array
        if isinstance(feat, tf.Tensor):
            feat_np = feat.numpy()
        else:
            feat_np = feat
            
        # Predict - now feat_np is already (1, 512)
        preds = model.predict([feat_np, input_seq], verbose=0)
        
        # Get next word
        pos = min(len(tokens) - 1, 18)
        next_id = np.argmax(preds[0, pos, :])
        
        # Stop if <END>
        if next_id == vocab['<END>']:
            break
            
        tokens.append(next_id)
    
    # Convert to Arabic
    words = []
    for token_id in tokens:
        if token_id in reverse_vocab:
            word = reverse_vocab[token_id]
            if word not in ['<START>', '<END>', '<PAD>']:
                words.append(word)
    
    return " ".join(words) if words else "لا يوجد وصف"

# Test on 5 random images
import random
test_images = list(features.keys())[:10]  # First 10 images
results = []

print("\n🤖 Testing on 5 images:")
for i, img_name in enumerate(test_images[:5], 1):
    caption = generate_caption(img_name)
    print(f"\n{i}. {img_name}")
    print(f"   Caption: {caption}")
    results.append({'image': img_name, 'caption': caption})

# Save results
import json
os.makedirs("results", exist_ok=True)
with open("results/test_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Results saved: results/test_results.json")

print("\n" + "="*70)
print("✅ TESTING COMPLETE!")
print("="*70)