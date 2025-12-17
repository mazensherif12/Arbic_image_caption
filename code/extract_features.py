"""
STEP 1: Extract features - Medium speed
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n" + "="*70)
print("🎯 STEP 1: EXTRACTING IMAGE FEATURES (FAST)")
print("="*70)

import pickle
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Paths
IMAGE_DIR = "data/Flickr8k_Dataset"
FEATURES_FILE = "data/vgg16_features.pickle"

# Check if already exists
if os.path.exists(FEATURES_FILE):
    print(f"✅ Features already exist: {FEATURES_FILE}")
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)
    print(f"Loaded {len(features)} images")
else:
    print("Loading VGG16 model...")
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    features = {}
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for i, img_file in enumerate(image_files[:5000]):  # Only 5000 for speed
        try:
            img_path = os.path.join(IMAGE_DIR, img_file)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            feat = model.predict(img_array, verbose=0)
            features[img_file.split('.')[0]] = feat.flatten()
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{min(5000, len(image_files))} images")
                
        except Exception as e:
            print(f"  Skipped {img_file}: {e}")
    
    # Save
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"✅ Saved features for {len(features)} images")

print("\n" + "="*70)
print("✅ STEP 1 COMPLETE!")
print("="*70)