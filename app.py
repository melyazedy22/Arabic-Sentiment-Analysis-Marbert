"""
Arabic Sentiment Analysis with MARBERT
Interactive Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
import sys
from io import BytesIO

# Add utils to path
if './utils' not in sys.path:
    sys.path.insert(0, './utils')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocessing import clean_arabic_text


# Page configuration
st.set_page_config(
    page_title="Arabic Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .negative {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .neutral {
        background-color: #f5f5f5;
        color: #616161;
        border: 2px solid #616161;
    }
    .positive {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path='./final_model'):
    """Load model and tokenizer with caching."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Load label mapping
        label_map_path = os.path.join(model_path, 'label_map.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                id2label = {int(k): v for k, v in label_map['id2label'].items()}
        else:
            id2label = model.config.id2label
        
        return model, tokenizer, device, id2label
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def predict_sentiment(text, model, tokenizer, device, id2label, clean_text=True):
    """Predict sentiment for a single text."""
    # Clean text if requested
    if clean_text:
        text = clean_arabic_text(text)
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get prediction
    pred_id = torch.argmax(probs, dim=-1).item()
    pred_label = id2label[pred_id]
    confidence = probs[0, pred_id].item()
    
    # Get all probabilities
    all_probs = {id2label[i]: probs[0, i].item() for i in range(len(id2label))}
    
    return pred_label, confidence, all_probs


def plot_probabilities(probs_dict):
    """Create a bar chart of class probabilities."""
    labels = list(probs_dict.keys())
    values = list(probs_dict.values())
    
    colors = {'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'}
    bar_colors = [colors.get(label, '#3498db') for label in labels]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🌙 Arabic Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by MARBERT - Multi-dialect Arabic BERT</div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading model...'):
        model, tokenizer, device, id2label = load_model()
    
    # Sidebar
    st.sidebar.title("📋 About")
    st.sidebar.info(
        """
        This application performs sentiment analysis on Arabic text 
        (Modern Standard Arabic and dialects).
        
        **Classes:**
        - 😡 Negative
        - 😐 Neutral
        - 😊 Positive
        
        **Model:** MARBERT (UBC-NLP)
        """
    )
    
    st.sidebar.title("⚙️ Settings")
    clean_input = st.sidebar.checkbox("Clean input text", value=True, help="Apply Arabic normalization")
    
    # Mode selection
    st.sidebar.title("🎯 Mode")
    mode = st.sidebar.radio("Select mode:", ["Single Prediction", "Batch Prediction", "Dashboard"])
    
    # Example sentences
    st.sidebar.title("💡 Examples")
    examples = {
        "Egyptian (Positive)": "الفيلم ده جميل جدا وممتع للغاية",
        "Gulf (Negative)": "المنتج سيء وما يستاهل الثمن",
        "MSA (Positive)": "هذا الكتاب رائع ومفيد جدا",
        "MSA (Negative)": "الخدمة سيئة جدا وغير مرضية",
        "Levantine (Neutral)": "الموضوع عادي مش كثير مهم"
    }
    
    selected_example = st.sidebar.selectbox("Try an example:", [""] + list(examples.keys()))
    
    # Main content
    if mode == "Single Prediction":
        st.header("🔍 Single Text Prediction")
        
        # Text input
        default_text = examples[selected_example] if selected_example else ""
        input_text = st.text_area(
            "Enter Arabic text:",
            value=default_text,
            height=100,
            placeholder="أدخل النص العربي هنا..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            predict_button = st.button("🚀 Analyze Sentiment", type="primary")
        
        if predict_button and input_text.strip():
            with st.spinner('Analyzing...'):
                # Predict
                pred_label, confidence, all_probs = predict_sentiment(
                    input_text,
                    model,
                    tokenizer,
                    device,
                    id2label,
                    clean_text=clean_input
                )
                
                # Display result
                st.markdown("### 📊 Results")
                
                # Sentiment box
                sentiment_classes = {
                    'negative': ('Negative', '😡', 'negative'),
                    'neutral': ('Neutral', '😐', 'neutral'),
                    'positive': ('Positive', '😊', 'positive')
                }
                
                label_name, emoji, css_class = sentiment_classes.get(pred_label, ('Unknown', '❓', 'neutral'))
                
                st.markdown(
                    f'<div class="sentiment-box {css_class}">{emoji} {label_name.upper()} (Confidence: {confidence:.2%})</div>',
                    unsafe_allow_html=True
                )
                
                # Probabilities
                st.markdown("### 📈 Probability Distribution")
                fig = plot_probabilities(all_probs)
                st.pyplot(fig)
                
                # Details
                with st.expander("📝 Details"):
                    st.write(f"**Original Text:** {input_text}")
                    if clean_input:
                        cleaned = clean_arabic_text(input_text)
                        st.write(f"**Cleaned Text:** {cleaned}")
                    st.write(f"**Predicted Label:** {pred_label}")
                    st.write(f"**Confidence:** {confidence:.4f}")
                    st.write("**All Probabilities:**")
                    for label, prob in all_probs.items():
                        st.write(f"  - {label}: {prob:.4f}")
        
        elif predict_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    elif mode == "Batch Prediction":
        st.header("📁 Batch Prediction")
        
        st.markdown("""
        Upload a CSV file with a column named **'text'** containing Arabic reviews.
        The app will predict sentiment for each row and allow you to download the results.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("❌ CSV must contain a column named 'text'")
                else:
                    st.success(f"✅ Loaded {len(df)} rows")
                    st.dataframe(df.head())
                    
                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        predictions = []
                        confidences = []
                        all_probs_list = []
                        
                        for i, text in enumerate(df['text']):
                            status_text.text(f"Processing {i+1}/{len(df)}...")
                            progress_bar.progress((i + 1) / len(df))
                            
                            pred_label, confidence, all_probs = predict_sentiment(
                                str(text),
                                model,
                                tokenizer,
                                device,
                                id2label,
                                clean_text=clean_input
                            )
                            
                            predictions.append(pred_label)
                            confidences.append(confidence)
                            all_probs_list.append(all_probs)
                        
                        # Add results to dataframe
                        df['predicted_sentiment'] = predictions
                        df['confidence'] = confidences
                        
                        for label in id2label.values():
                            df[f'prob_{label}'] = [probs[label] for probs in all_probs_list]
                        
                        status_text.text("✅ Prediction complete!")
                        
                        st.markdown("### 📊 Results")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="sentiment_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Statistics
                        st.markdown("### 📈 Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        sentiment_counts = pd.Series(predictions).value_counts()
                        
                        with col1:
                            neg_count = sentiment_counts.get('negative', 0)
                            st.metric("😡 Negative", neg_count, f"{neg_count/len(df)*100:.1f}%")
                        
                        with col2:
                            neu_count = sentiment_counts.get('neutral', 0)
                            st.metric("😐 Neutral", neu_count, f"{neu_count/len(df)*100:.1f}%")
                        
                        with col3:
                            pos_count = sentiment_counts.get('positive', 0)
                            st.metric("😊 Positive", pos_count, f"{pos_count/len(df)*100:.1f}%")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    elif mode == "Dashboard":
        st.header("📊 Dashboard")
        
        st.markdown("""
        Upload a CSV file with predictions to visualize sentiment distribution.
        Use the **Batch Prediction** mode first to generate predictions.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV with predictions", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check for required columns
                if 'predicted_sentiment' in df.columns:
                    sentiment_col = 'predicted_sentiment'
                elif 'sentiment' in df.columns:
                    sentiment_col = 'sentiment'
                else:
                    st.error("❌ CSV must contain 'predicted_sentiment' or 'sentiment' column")
                    st.stop()
                
                st.success(f"✅ Loaded {len(df)} predictions")
                
                # Sentiment distribution
                sentiment_counts = df[sentiment_col].value_counts()
                
                # Pie chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                colors = {
                    'negative': '#e74c3c',
                    'neutral': '#95a5a6',
                    'positive': '#2ecc71'
                }
                
                pie_colors = [colors.get(label, '#3498db') for label in sentiment_counts.index]
                
                ax1.pie(
                    sentiment_counts.values,
                    labels=sentiment_counts.index,
                    autopct='%1.1f%%',
                    colors=pie_colors,
                    startangle=90,
                    textprops={'fontsize': 11, 'fontweight': 'bold'}
                )
                ax1.set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
                
                # Bar chart
                bars = ax2.bar(
                    sentiment_counts.index,
                    sentiment_counts.values,
                    color=[colors.get(label, '#3498db') for label in sentiment_counts.index],
                    edgecolor='black',
                    linewidth=1.5
                )
                ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
                ax2.set_title('Sentiment Counts', fontsize=12, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontweight='bold'
                    )
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data sample
                st.markdown("### 📝 Data Sample")
                st.dataframe(df.head(20))
                
                # Statistics
                st.markdown("### 📊 Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Samples", len(df))
                
                with col2:
                    neg_pct = (sentiment_counts.get('negative', 0) / len(df)) * 100
                    st.metric("Negative %", f"{neg_pct:.1f}%")
                
                with col3:
                    neu_pct = (sentiment_counts.get('neutral', 0) / len(df)) * 100
                    st.metric("Neutral %", f"{neu_pct:.1f}%")
                
                with col4:
                    pos_pct = (sentiment_counts.get('positive', 0) / len(df)) * 100
                    st.metric("Positive %", f"{pos_pct:.1f}%")
                
                # Confidence analysis (if available)
                if 'confidence' in df.columns:
                    st.markdown("### 🎯 Confidence Analysis")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(df['confidence'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                    ax.axvline(df['confidence'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
                    ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                    ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.write(f"**Average Confidence:** {df['confidence'].mean():.4f}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #7f8c8d;'>
        Built with ❤️ using Streamlit and Hugging Face Transformers
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
