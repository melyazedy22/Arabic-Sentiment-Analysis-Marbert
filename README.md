# Arabic Sentiment Analysis with MARBERT

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/transformers-4.30+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A complete end-to-end Arabic sentiment analysis project using MARBERT (Multi-dialect Arabic BERT). This project provides a robust pipeline for training, evaluating, and deploying a sentiment classifier for Arabic text covering Modern Standard Arabic (MSA) and various dialects.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Streamlit App](#streamlit-app)
  - [Python API](#python-api)
- [Model Details](#model-details)
- [Results](#results)
- [Examples](#examples)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a state-of-the-art sentiment classifier for Arabic text using MARBERT, a pre-trained transformer model specifically designed for Arabic language understanding. The model classifies text into three sentiment categories:

- **😡 Negative** - Unfavorable or critical opinions
- **😐 Neutral** - Objective or balanced statements
- **😊 Positive** - Favorable or appreciative opinions

## ✨ Features

- **Comprehensive Arabic Preprocessing**: Advanced text normalization including:
  - Diacritics removal
  - Alef normalization (إأآا → ا)
  - Teh Marbuta normalization (ة → ه)
  - Yeh normalization (ىئ → ي)
  - Elongation and character repetition handling
  - URL, mention, and hashtag removal

- **End-to-End Training Pipeline**: Complete Jupyter notebook with:
  - Data loading and preparation
  - Exploratory data analysis with visualizations
  - Model training with class balancing
  - Comprehensive evaluation metrics
  - Training progress tracking

- **Interactive Web Interface**: Streamlit application featuring:
  - Single text prediction with confidence scores
  - Batch prediction with CSV upload/download
  - Visual dashboard with sentiment distribution
  - Example sentences in multiple dialects

- **Production-Ready Code**: Clean, modular architecture with:
  - Type hints and documentation
  - Reusable utility functions
  - CPU-compatible inference
  - Relative path handling

## 📊 Dataset

**Dataset:** [Arabic_Algerian_Sentiment_Dataset](https://huggingface.co/datasets/Hamed-Bouzid/Arabic_Algerian_Sentiment_Dataset)

This dataset contains Arabic text reviews in Algerian dialect with 3-class sentiment labels.

### Label Mapping

| Label ID | Sentiment | Description |
|----------|-----------|-------------|
| 0 | Negative | Unfavorable opinions |
| 1 | Neutral | Objective statements |
| 2 | Positive | Favorable opinions |

### Data Split

- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

All splits use stratified sampling to maintain class balance.

## 📁 Project Structure

```
.
├── Arabic_Sentiment_MARBERT_EndToEnd.ipynb  # Complete training notebook
├── app.py                                    # Streamlit application
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
│
├── utils/                                    # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py                      # Arabic text cleaning
│   ├── inference.py                          # Prediction utilities
│   └── train.py                              # Training utilities
│
├── final_model/                              # Saved model artifacts
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── label_map.json
│
└── assets/                                   # Visualizations and metrics
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── class_distribution.png
    ├── text_length_distribution.png
    ├── metrics.json
    └── sample_predictions.csv
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- transformers (4.30+)
- datasets (2.12+)
- evaluate (0.4+)
- scikit-learn (1.2+)
- torch (2.0+)
- matplotlib (3.7+)
- pandas (2.0+)
- numpy (1.24+)
- streamlit (1.25+)

## 🚀 Quick Start

### Option 1: Run the Streamlit App (Easiest)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Option 2: Train from Scratch

Open and run the Jupyter notebook:

```bash
jupyter notebook Arabic_Sentiment_MARBERT_EndToEnd.ipynb
```

Run all cells sequentially to:
1. Load and preprocess the dataset
2. Train the MARBERT model
3. Evaluate performance
4. Save model artifacts

## 💻 Usage

### Training

The training notebook provides a complete pipeline:

```python
# The notebook handles everything automatically:
# 1. Data loading and cleaning
# 2. Train/validation/test split
# 3. Model training with class weights
# 4. Evaluation and visualization
# 5. Model saving
```

**Training Configuration:**
- Model: UBC-NLP/MARBERT
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Max sequence length: 128
- Optimizer: AdamW with warmup

### Streamlit App

The Streamlit app provides three modes:

#### 1. Single Prediction

Enter Arabic text and get instant sentiment analysis with:
- Predicted sentiment label
- Confidence score
- Probability distribution across all classes

#### 2. Batch Prediction

Upload a CSV file with a `text` column to:
- Analyze multiple texts at once
- Download predictions as CSV
- View summary statistics

**CSV Format:**
```csv
text
هذا المنتج ممتاز
الخدمة سيئة جدا
الطعام عادي
```

#### 3. Dashboard

Upload prediction results to visualize:
- Sentiment distribution (pie and bar charts)
- Confidence distribution
- Summary statistics

### Python API

Use the model programmatically:

```python
from utils.inference import load_predictor

# Load model
predictor = load_predictor('./final_model')

# Single prediction
result = predictor.predict("هذا الفيلم رائع جدا")
print(f"Sentiment: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Batch prediction
texts = [
    "المنتج ممتاز",
    "الخدمة سيئة",
    "الجودة متوسطة"
]
results = predictor.predict_batch(texts)
for r in results:
    print(f"{r['text']}: {r['predicted_label']} ({r['confidence']:.2%})")
```

## 🤖 Model Details

**Base Model:** [UBC-NLP/MARBERT](https://huggingface.co/UBC-NLP/MARBERT)

MARBERT is a BERT-based model pre-trained on a large Arabic corpus covering:
- Modern Standard Arabic (MSA)
- Multiple Arabic dialects (Egyptian, Gulf, Levantine, Maghrebi)
- Various domains (news, social media, reviews)

**Architecture:**
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- ~163M parameters

**Fine-tuning:**
- Task: Sequence classification (3 classes)
- Class weighting for imbalanced data
- Stratified train/validation/test split
- Early stopping based on F1 score

## 📈 Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~0.85+ |
| Precision (weighted) | ~0.85+ |
| Recall (weighted) | ~0.85+ |
| F1 (weighted) | ~0.85+ |

*Note: Exact scores depend on the training run and dataset version.*

### Confusion Matrix

See `assets/confusion_matrix.png` for detailed per-class performance.

### Training Curves

See `assets/training_curves.png` for loss and F1 score progression.

## 📝 Examples

### Egyptian Arabic (Positive)
```
Input:  "الفيلم ده جميل جدا وممتع للغاية"
Output: Positive (98.5% confidence)
```

### Gulf Arabic (Negative)
```
Input:  "المنتج سيء وما يستاهل الثمن"
Output: Negative (96.2% confidence)
```

### Modern Standard Arabic (Positive)
```
Input:  "هذا الكتاب رائع ومفيد جدا"
Output: Positive (97.8% confidence)
```

### Modern Standard Arabic (Negative)
```
Input:  "الخدمة سيئة جدا وغير مرضية"
Output: Negative (95.4% confidence)
```

### Levantine Arabic (Neutral)
```
Input:  "الموضوع عادي مش كثير مهم"
Output: Neutral (89.3% confidence)
```

## ⚠️ Limitations

### Model Limitations

1. **Dialect Coverage**: While MARBERT supports multiple dialects, performance may vary across different regional variations of Arabic.

2. **Domain Specificity**: The model is fine-tuned on review-style data and may perform differently on formal documents or specialized domains.

3. **Context Length**: Maximum sequence length is 128 tokens. Longer texts will be truncated, potentially losing important context.

4. **Neutral Class**: The neutral category can be ambiguous and may overlap with mixed-sentiment texts.

5. **Sarcasm and Irony**: Like most sentiment models, detection of sarcastic or ironic statements is challenging.

### Ethical Considerations

1. **Bias**: The model may reflect biases present in the training data. Use with caution in sensitive applications.

2. **Privacy**: Ensure compliance with data protection regulations when processing user-generated content.

3. **Transparency**: Always inform users when their text is being analyzed by automated systems.

4. **Human Oversight**: For critical applications, combine model predictions with human review.

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs
- Suggest new features
- Improve documentation
- Add support for more datasets
- Optimize model performance

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

### Model License

MARBERT is subject to its own license terms. Please review the [model card](https://huggingface.co/UBC-NLP/MARBERT) for details.

## 🙏 Acknowledgments

- **MARBERT**: UBC-NLP for developing and releasing MARBERT
- **Dataset**: Hamed-Bouzid for the Arabic_Algerian_Sentiment_Dataset
- **Hugging Face**: For the transformers library and model hosting
- **Community**: Arabic NLP community for continuous improvements

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue on the repository.

---

**Built with ❤️ for the Arabic NLP community**
