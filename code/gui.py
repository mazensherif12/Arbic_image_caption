"""
STEP 5: SIMPLE GUI - Show results with image display fix
"""
import tkinter as tk
from PIL import Image, ImageTk
import os
import json

print("\n" + "="*70)
print("🎯 STEP 5: SIMPLE GUI")
print("="*70)

# Check results
results_path = "results/test_results.json"
if not os.path.exists(results_path):
    print(f"❌ No test results found at: {results_path}")
    print("Run: python 4_test.py")
    exit()

print(f"Loading results from: {results_path}")

# Load results
with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f"✅ Loaded {len(results)} results")

# Debug: Show what's in results
print("\n🔍 Sample results (first 3):")
for i, result in enumerate(results[:3]):
    print(f"  {i+1}. Image: '{result['image']}'")
    print(f"     Caption: '{result['caption'][:50]}...'")

# Check image directory
image_dir = "data/Flickr8k_Dataset"
print(f"\n📁 Checking image directory: {image_dir}")
if os.path.exists(image_dir):
    files = os.listdir(image_dir)[:3]  # Show first 3 files
    print(f"   Found directory, sample files: {files}")
else:
    print(f"   ❌ Directory not found!")

# Check a specific image
if len(results) > 0:
    test_image = results[0]['image']
    possible_paths = [
        f"{image_dir}/{test_image}",
        f"{image_dir}/{test_image}.jpg",
        f"{image_dir}/{test_image}.jpeg",
        f"{image_dir}/{test_image}.png",
    ]
    print(f"\n🔍 Looking for image '{test_image}':")
    for path in possible_paths:
        exists = os.path.exists(path)
        print(f"   {path}: {'✅ Found' if exists else '❌ Not found'}")

class SimpleViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Caption Results")
        self.root.geometry("900x700")
        
        self.results = results
        self.current = 0
        
        print(f"\n🚀 Creating GUI with {len(self.results)} results...")
        self.create_widgets()
        self.show_result(0)
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Arabic Image Captioning Results", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=(10, 0))
        
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left side - Image
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 20))
        
        img_title = tk.Label(left_frame, text="Image Preview", font=('Arial', 12, 'bold'))
        img_title.pack()
        
        # Image container with border
        img_container = tk.Frame(left_frame, bg='gray', relief='sunken', bd=2)
        img_container.pack(fill='both', expand=True, pady=(5, 0))
        
        self.img_label = tk.Label(img_container, bg='white')
        self.img_label.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Right side - Caption
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        
        caption_title = tk.Label(right_frame, text="Generated Arabic Caption", 
                                font=('Arial', 12, 'bold'))
        caption_title.pack()
        
        # Caption text with scrollbar
        caption_frame = tk.Frame(right_frame)
        caption_frame.pack(fill='both', expand=True, pady=(5, 0))
        
        # Add a border to caption frame
        caption_border = tk.Frame(caption_frame, bg='gray', relief='sunken', bd=2)
        caption_border.pack(fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(caption_border)
        scrollbar.pack(side='right', fill='y')
        
        self.caption_text = tk.Text(caption_border, height=8, width=40, 
                                   font=('Arial', 16), wrap='word',
                                   yscrollcommand=scrollbar.set,
                                   bg='white', relief='flat')
        self.caption_text.pack(fill='both', expand=True, padx=2, pady=2)
        scrollbar.config(command=self.caption_text.yview)
        
        # Info panel
        info_frame = tk.Frame(self.root)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        self.info_label = tk.Label(info_frame, text="", font=('Arial', 11))
        self.info_label.pack()
        
        # Navigation
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=(0, 10))
        
        self.prev_btn = tk.Button(nav_frame, text="⬅ Previous", command=self.prev, 
                                 width=15, font=('Arial', 12))
        self.prev_btn.pack(side='left', padx=5)
        
        self.next_btn = tk.Button(nav_frame, text="Next ➡", command=self.next,
                                 width=15, font=('Arial', 12))
        self.next_btn.pack(side='left', padx=5)
        
        # Status bar
        status_frame = tk.Frame(self.root, relief='sunken', bd=1)
        status_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Ready", font=('Arial', 10))
        self.status_label.pack(side='left', padx=5, pady=2)
        
        # Update button states
        self.update_nav_buttons()
    
    def find_image_file(self, image_name):
        """Try to find image file with various extensions and paths"""
        # Possible image extensions
        extensions = ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG']
        
        # Possible directories
        possible_dirs = [
            "data/Flickr8k_Dataset",
            "Flickr8k_Dataset",
            "data/images",
            "images",
            "data",
            "."
        ]
        
        for directory in possible_dirs:
            if os.path.exists(directory):
                for ext in extensions:
                    img_path = os.path.join(directory, f"{image_name}{ext}")
                    if os.path.exists(img_path):
                        print(f"   ✅ Found image at: {img_path}")
                        return img_path
        
        print(f"   ❌ Could not find image for: {image_name}")
        return None
    
    def show_result(self, idx):
        if 0 <= idx < len(self.results):
            self.current = idx
            result = self.results[idx]
            
            print(f"\n📊 Showing result {idx+1}/{len(self.results)}")
            print(f"   Image name: {result['image']}")
            print(f"   Caption: {result['caption']}")
            
            # Update info
            info_text = f"Image {idx+1} of {len(self.results)} | Name: {result['image']}"
            self.info_label.config(text=info_text)
            
            # Update caption
            self.caption_text.config(state='normal')
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(1.0, result['caption'])
            self.caption_text.config(state='disabled')
            
            # Try to load image
            img_path = self.find_image_file(result['image'])
            
            if img_path and os.path.exists(img_path):
                try:
                    print(f"   Loading image: {img_path}")
                    img = Image.open(img_path)
                    
                    # Calculate size to fit in container
                    container_width = 400
                    container_height = 400
                    
                    # Get image size
                    img_width, img_height = img.size
                    
                    # Calculate aspect ratio
                    ratio = min(container_width/img_width, container_height/img_height)
                    new_size = (int(img_width * ratio), int(img_height * ratio))
                    
                    # Resize
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    
                    # Update image label
                    self.img_label.config(image=photo, text="")
                    self.img_label.image = photo  # Keep reference
                    
                    self.status_label.config(text=f"✓ Image loaded successfully", fg='green')
                    print(f"   ✅ Image displayed successfully")
                    
                except Exception as e:
                    error_msg = f"Error loading image: {str(e)}"
                    print(f"   ❌ {error_msg}")
                    self.img_label.config(text=error_msg[:50], font=('Arial', 10), fg='red')
                    self.status_label.config(text=error_msg[:50], fg='red')
            else:
                error_msg = f"Image file not found: {result['image']}"
                print(f"   ❌ {error_msg}")
                self.img_label.config(
                    text=f"Image not found:\n{result['image']}",
                    font=('Arial', 12), 
                    fg='orange'
                )
                self.status_label.config(text=error_msg, fg='orange')
            
            # Update navigation buttons
            self.update_nav_buttons()
    
    def update_nav_buttons(self):
        """Enable/disable navigation buttons based on current position"""
        self.prev_btn.config(state='normal' if self.current > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current < len(self.results) - 1 else 'disabled')
    
    def prev(self):
        if self.current > 0:
            print(f"\n⬅ Navigating to previous image...")
            self.show_result(self.current - 1)
    
    def next(self):
        if self.current < len(self.results) - 1:
            print(f"\n➡ Navigating to next image...")
            self.show_result(self.current + 1)

def main():
    print("\n🚀 Starting GUI application...")
    try:
        root = tk.Tk()
        app = SimpleViewer(root)
        print("✅ GUI created successfully")
        print("⏳ Entering main loop...")
        root.mainloop()
        print("✅ GUI closed")
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()