import tensorflow as tf
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Input, Embedding, Dropout, Dense, LSTM, Bidirectional,
    GlobalAveragePooling2D, Concatenate
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

VGG16_EMBEDDING_SIZE = 512
MAXLEN = 20


class Encoder(tf.keras.Model):
    """
    Encoder using VGG16 for feature extraction
    """
    def __init__(self):
        super(Encoder, self).__init__()

        base_vgg = VGG16(weights='imagenet', include_top=False)
        for layer in base_vgg.layers:
            layer.trainable = False

        self.encoder = tf.keras.Sequential([
            base_vgg,
            GlobalAveragePooling2D()
        ])

    def call(self, img):
        return self.encoder(tf.expand_dims(img, axis=0))


class Decoder():
    """
    MEDIUM-SIZED DECODER
    - Single LSTM with attention
    - Medium embedding size
    - Good for Arabic vocabulary
    """
    
    def __init__(self, vocab_size):
        print("\n" + "="*70)
        print("🏗️  BUILDING MEDIUM MODEL")
        print("="*70)
        
        self.vocab_size = vocab_size
        
        # Model dimensions
        self.image_embedding_size = 256    
        self.text_embedding_size = 256     
        self.lstm_units = 256              
        self.dense_units = 256

        self.input_image = Input(shape=(VGG16_EMBEDDING_SIZE,), name='image_input')
        self.input_text = Input(shape=(MAXLEN-1,), name='text_input')

        # Image processing
        self.image_dense = Dense(
            units=self.image_embedding_size,
            activation='relu',
            kernel_initializer='he_normal',
            name='image_dense'
        )
        self.image_dropout = Dropout(0.3, name='image_dropout')
        
        # Text embedding
        self.text_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.text_embedding_size,
            embeddings_initializer='glorot_uniform',
            mask_zero=True,
            name='text_embedding'
        )
        
        # LSTM layer
        self.lstm = LSTM(
            units=self.lstm_units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            name='lstm'
        )
        
        # Output layers
        self.output_dense1 = Dense(
            units=self.dense_units,
            activation='relu',
            kernel_initializer='he_normal',
            name='output_dense1'
        )
        self.output_dropout = Dropout(0.3, name='output_dropout')
        
        # Final output
        self.output_softmax = Dense(
            units=vocab_size,
            activation="softmax",
            kernel_initializer='glorot_uniform',
            name='output_softmax'
        )

        print(f"✓ Vocabulary: {vocab_size:,} words")
        print(f"✓ Embedding Size: {self.text_embedding_size}D")
        print(f"✓ LSTM Units: {self.lstm_units}")
        print(f"✓ Architecture: Simple LSTM")
        print("="*70 + "\n")

    def get_model(self):
        # Process image
        img_features = self.image_dense(self.input_image)
        img_features = self.image_dropout(img_features)
        
        # Repeat image features for each time step
        img_features_expanded = tf.keras.layers.RepeatVector(MAXLEN - 1)(img_features)
        
        # Process text
        text_embeddings = self.text_embedding(self.input_text)
        
        # Concatenate image and text features
        combined = Concatenate(axis=-1)([text_embeddings, img_features_expanded])
        
        # LSTM processing
        lstm_out = self.lstm(combined)
        
        # Output layers
        output = self.output_dense1(lstm_out)
        output = self.output_dropout(output)
        
        # Final softmax
        final_output = self.output_softmax(output)

        # Build model
        decoder_model = tf.keras.Model(
            inputs=[self.input_image, self.input_text],
            outputs=final_output,
            name='arabic_caption_decoder'
        )
        
        # Summary
        decoder_model.summary(line_length=70)
        
        return decoder_model