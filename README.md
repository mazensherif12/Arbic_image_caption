 Arabic Image Caption Generator

This project generates **Arabic captions** for images using deep learning techniques. It is designed to help bridge the gap between image understanding and the Arabic language, making image content more accessible to Arabic speakers.

## Features

- Generates natural Arabic captions for images.
- Uses **deep learning models** for image-to-text translation.
- Trained on the **Flickr8k dataset**, a popular dataset for image captioning.

## Dataset

The project uses the **Flickr8k dataset**, which contains 8,000 images each annotated with 5 English captions. These captions are used as a base for generating Arabic captions.

- **Flickr8k dataset link**: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
- Images are preprocessed and captions are translated to Arabic for training.

## Requirements

- Python 3.8+
- TensorFlow / PyTorch (depending on implementation)
- Numpy, Pandas, Matplotlib
- nltk (for text preprocessing)
- OpenCV / PIL (for image preprocessing)

Install dependencies using pip:

```bash
pip install -r requirements.txt
