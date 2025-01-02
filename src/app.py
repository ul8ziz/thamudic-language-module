import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from core.inference_engine import InferenceEngine
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_inference_engine():
    """
    تحميل محرك التنبؤ
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'models')
        
        if not os.path.exists(model_dir):
            st.error("نموذج التعرف غير موجود. يرجى تدريب النموذج أولاً.")
            return None
            
        engine = InferenceEngine(model_dir)
        return engine
        
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None

def main():
    # Set page config
    st.set_page_config(
        page_title="نظام التعرف على النقوش الثمودية",
        page_icon="🔍",
        layout="wide"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            direction: rtl;
            text-align: right;
        }
        .css-1d391kg {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f7f7f7;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("نظام التعرف على النقوش الثمودية")
    st.markdown("""
        هذا النظام يقوم بالتعرف على الحروف الثمودية في الصور باستخدام تقنيات الذكاء الاصطناعي.
        قم برفع صورة للنقش الثمودي وسيقوم النظام بتحليلها وتحديد الحروف الموجودة فيها.
    """)
    
    # Load model
    engine = load_inference_engine()
    if engine is None:
        return
        
    # File uploader
    uploaded_file = st.file_uploader(
        "قم برفع صورة للنقش الثمودي",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        try:
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("الصورة الأصلية")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            # Save temporary file
            temp_path = "temp_image.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Get predictions
            with st.spinner('جاري تحليل الصورة...'):
                predictions = engine.predict(temp_path)
            
            # Display results
            with col2:
                st.subheader("نتائج التحليل")
                for pred in predictions:
                    confidence = pred['confidence'] * 100
                    st.markdown(f"""
                        <div class='css-1d391kg'>
                            <h3>الحرف: {pred['letter']}</h3>
                            <p>نسبة الثقة: {confidence:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء تحليل الصورة: {str(e)}")
            logging.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
