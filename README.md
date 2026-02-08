# Arabic Sentiment Analysis with MARBERT

## 📌 Project Overview
An end-to-end Sentiment Analysis project for Arabic text (MSA and Dialects) using the state-of-the-art **MARBERT** model. The project includes a training notebook, a testing notebook, and an interactive **Streamlit** web application.

## 📊 Dataset
**Source:** [HARD Dataset (Hotel Arabic Reviews Dataset)](https://huggingface.co/datasets/Elnagara/hard)
- **Size:** ~105,000 Reviews
- **Content:** Hotel reviews in Arabic.
- **Classes:** Balanced dataset with Negative, Neutral, and Positive reviews.

## 🛠️ Model Details
- **Architecture:** `UBC-NLP/MARBERT` (BertForSequenceClassification)
- **Fine-tuned:** On the HARD dataset for 3-epoch training.
- **Performance:** High accuracy in distinguishing sentiment polarity.

### 🏷️ Label Mapping (Corrected)
After extensive verification, the final label mapping for this model is:

| ID | Label | Description |
| :--- | :--- | :--- |
| **0** | **Negative** | Strong negative sentiment (e.g., "Very bad", "Hated it") |
| **1** | **Positive** | Positive sentiment (e.g., "Great", "Excellent") |
| **2** | **Neutral** | Neutral or ordinary experiences (e.g., "Ordinary", "Average") |

> **Note:** Earlier versions had incorrect mappings. Please ensure your `config.json` matches this valid mapping.

---

## 🚀 Installation & Setup

### 1. Requirements
This project requires specific library versions to avoid compatibility issues.
```bash
pip install -r requirements.txt
```
**Key Dependencies:**
- `transformers==4.37.2`
- `accelerate==0.27.2`
- `torch>=2.0.0`
- `streamlit`

### 2. Running the App
To start the interactive web interface:
```bash
streamlit run app.py
```

### 3. Testing the Model
You can test the model programmatically using the provided notebook:
- Open `Test_model.ipynb` in Jupyter/VSCode.
- Run the cells to see predictions on sample text.

---

## 📂 Project Structure
- **`app.py`**: The main Streamlit application script.
- **`final_model/`**: Contains the fine-tuned model files (`model.safetensors`, `config.json`, etc.).
- **`utils/`**: Helper functions for text preprocessing (`clean_arabic_text`).
- **`Test_model.ipynb`**: Notebook for debugging and verification.
- **`Arabic_Sentiment_MARBERT_EndToEnd.ipynb`**: The original training notebook.
- **`requirements.txt`**: Locked dependency versions.

---

## 🔄 Recent Updates & Fixes
**Latest Version (v1.1)**
- **✅ Fixed Import Errors:** Resolved `ImportError: cannot import name 'BertForSequenceClassification'` by correctly handling dynamic loading in `app.py` and `Test_model.ipynb`.
- **✅ Label Mapping Correction:** Corrected the confusion matrix mapping. Previous iterations incorrectly labeled "Neutral" as Negative. The new mapping (0=Neg, 1=Pos, 2=Neu) has been verified with test samples.
- **✅ Environment Stability:** Downgraded `transformers` to `4.37.2` to resolve stability issues with the latest Hugging Face release.
- **✅ Cleanup:** Removed temporary debug scripts and redundant backup files (`final_model.zip`) to save space.

---

## 📝 Usage Example
**Input:** "خدمة رائعة جدا" (Very great service)
**Output:** `Positive` (Confidence: 99%)

**Input:** "تجربة سيئة ولا أنصح بها" (Bad experience, do not recommend)
**Output:** `Negative` (Confidence: 95%)

**Input:** "مستوى عادي" (Ordinary level)
**Output:** `Neutral` (Confidence: 85%)

---
## 🔗 Resources (Model, Assets & Live Demo)

- **📦 Full Project Package (Model, Assets & Live Demo)**  
  <[DRIVE Link](https://drive.google.com/drive/folders/1t8L_vb6YIbjcHqgNcuwzGrIjWI0wonBi?usp=drive_link)>

> This link contains the fine-tuned MARBERT model, all utility files, assets, and access to the live Streamlit demo.
---
*Created by Mahmoud Elyazedy for Graduation Project*
