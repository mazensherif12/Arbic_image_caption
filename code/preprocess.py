"""
STEP 2: PREPROCESS DATA - Fixed syntax error
"""
import os
import re
import pickle

print("\n" + "="*70)
print("🎯 STEP 2: PREPROCESSING DATA")
print("="*70)

DATA_FILE = "data/Flickr8k_text/Flickr8k.arabic.full.txt"

def simple_clean(text):
    """Simple cleaning: keep Arabic, remove English only"""
    if not text or not isinstance(text, str):
        return ""
    
    # ONLY remove English letters and numbers
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    
    # Keep everything else
    return text.strip()

# Read file
print("Reading captions...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Parse
captions = {}
valid_lines = 0

for i, line in enumerate(lines):
    if '\t' in line:
        parts = line.split('\t')
        if len(parts) >= 2:
            img_name = parts[0].strip()
            raw_caption = parts[1].strip()
            
            # Simple cleaning
            caption = simple_clean(raw_caption)
            
            # Check if we still have text
            if caption and len(caption) > 3:  # At least 3 characters
                img_base = img_name.split('.')[0]
                if img_base not in captions:
                    captions[img_base] = []
                captions[img_base].append(f"<START> {caption} <END>")
                valid_lines += 1
    
    # Progress
    if (i + 1) % 2000 == 0:
        print(f"  Processed {i+1}/{len(lines)} lines")

print(f"\n✅ Valid caption lines: {valid_lines}")
print(f"✅ Images with captions: {len(captions)}")

# Show sample
if captions:
    sample_img = list(captions.keys())[0]
    raw_line = lines[0]
    raw_parts = raw_line.split('\t')
    raw_caption = raw_parts[1] if len(raw_parts) > 1 else 'N/A'
    
    print(f"\n📝 Sample:")
    print(f"   Image: {sample_img}")
    print(f"   Raw: {raw_caption.strip()}")
    print(f"   Cleaned: {captions[sample_img][0]}")

# Create vocabulary
print("\nCreating vocabulary...")
words = set()
for img_caps in captions.values():
    for cap in img_caps:
        for word in cap.split():
            if word not in ['<START>', '<END>', '<PAD>']:
                words.add(word)

print(f"Unique words found: {len(words)}")

# Build vocab
word_list = sorted(words)
vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}

# Limit vocabulary size for speed
MAX_VOCAB = 8000
if len(word_list) > MAX_VOCAB:
    print(f"Limiting vocabulary to {MAX_VOCAB} most common words")
    word_list = word_list[:MAX_VOCAB]

for i, word in enumerate(word_list, 3):
    vocab[word] = i

print(f"✅ Final vocabulary size: {len(vocab)} words")

# Save
with open("vocabulary.pkl", 'wb') as f:
    pickle.dump(vocab, f)
print(f"✅ Saved to: vocabulary.pkl")

# Save captions
with open("captions.pkl", 'wb') as f:
    pickle.dump(captions, f)
print(f"✅ Saved captions to: captions.pkl")

print("\n" + "="*70)
print("✅ STEP 2 COMPLETE!")
print("="*70)