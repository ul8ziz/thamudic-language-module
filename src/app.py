import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
from predict import ThamudicPredictor
import os
from pathlib import Path

# Initialize predictor
MODEL_PATH = 'output/models/best_model.pt'
LABEL_MAPPING_PATH = 'output/label_mapping.json'

@st.cache_resource
def load_predictor():
    return ThamudicPredictor(MODEL_PATH, LABEL_MAPPING_PATH)

def main():
    st.title("نظام التعرف على النقوش الثمودية")
    st.write("قم برفع صورة للنقش الثمودي للتعرف عليه")

    # Initialize predictor
    try:
        predictor = load_predictor()
        st.success("تم تحميل النموذج بنجاح!")
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("اختر صورة للنقش", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded image
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "temp_inscription.png"
        
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display original image
        st.subheader("الصورة المدخلة:")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Create visualization path
        vis_path = temp_dir / "visualization.png"
            
        # Process image and get predictions
        try:
          
            # Create visualization path
            vis_path = temp_dir / "visualization.png"
            
            # Get predictions
            predictions = predictor.predict_inscription(str(temp_path), str(vis_path))
            
            # Display results
            st.subheader("النتائج:")
            
            # Display visualization
            if vis_path.exists():
                st.image(str(vis_path), use_column_width=True)
            
            # Display text results
            st.write("الحروف المكتشفة:")
            for char, conf in predictions:
                st.write(f"الحرف: {char}, الثقة: {conf:.2f}")
            
            # Display full text
            full_text = "".join([char for char, _ in predictions])
            st.subheader("النص الكامل:")
            st.write(full_text)
            
        except Exception as e:
            st.error(f"خطأ في معالجة الصورة: {str(e)}")
        
        # Cleanup temporary files
        if temp_path.exists():
            temp_path.unlink()
        if vis_path.exists():
            vis_path.unlink()

if __name__ == '__main__':
    main()
