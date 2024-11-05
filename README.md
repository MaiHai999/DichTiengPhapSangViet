# Dịch Tiếng Pháp Sang Việt

This project provides a translation service from French to Vietnamese using advanced natural language processing techniques. It employs the `sep2seq` model for translation, utilizes `gensim` for word embeddings with Word2Vec, and leverages `underthesesea` for processing Vietnamese text.

## Features

- **French to Vietnamese Translation**: Translate French text into Vietnamese with high accuracy.
- **Word Embedding**: Use Word2Vec for generating word embeddings to enhance translation quality.
- **Vietnamese Language Processing**: Process and handle Vietnamese text effectively using `underthesea`.
- **Customizable**: Easily adaptable for different language pairs or extensions.

## Technologies Used

- **Python**: The primary programming language used for implementation.
- **sep2sel**: A translation model for efficient language translation.
- **gensim**: A library for unsupervised topic modeling and natural language processing, specifically for Word2Vec.
- **unthesesa**: A library designed to handle and preprocess Vietnamese text, ensuring proper tokenization and normalization.


## How It Works

1. **Data Collection**: Collect parallel datasets of French and Vietnamese sentences for training and evaluation.
2. **Text Preprocessing**: Use `unthesesa` to preprocess and normalize Vietnamese text to ensure accurate tokenization.
3. **Model Training**: Implement the `sep2sel` model to train on the collected datasets for effective translation.
4. **Word Embedding**: Utilize `gensim`'s Word2Vec to create word embeddings that enhance translation context and accuracy.
5. **Translation Service**: Provide a function to translate French text into Vietnamese using the trained model.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/dich_tieng_phap_sang_viet.git
   cd dich_tieng_phap_sang_viet


