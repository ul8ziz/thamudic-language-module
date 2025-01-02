import os
import json
import logging
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

def load_model_and_mapping():
    """
    تحميل النموذج وخريطة التصنيفات
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'best_model.h5')
        model_arch_path = os.path.join(base_dir, 'models', 'best_model_architecture.json')
        model_weights_path = os.path.join(base_dir, 'models', 'best_model_weights.h5')
        mapping_path = os.path.join(base_dir, 'data', 'letters', 'letter_mapping.json')
        
        if not os.path.exists(mapping_path):
            st.error("ملف خريطة الحروف غير موجود.")
            return None, None
            
        # Try loading full model first
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                st.success("تم تحميل النموذج بنجاح!")
            except Exception as e:
                logging.error(f"Error loading full model: {str(e)}")
                model = None
        else:
            model = None
            
        # If full model loading failed, try loading from architecture and weights
        if model is None and os.path.exists(model_arch_path) and os.path.exists(model_weights_path):
            try:
                # Load architecture
                with open(model_arch_path, 'r') as f:
                    model_json = f.read()
                model = tf.keras.models.model_from_json(model_json)
                
                # Load weights
                model.load_weights(model_weights_path)
                
                # Compile model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                st.success("تم تحميل النموذج من الهيكل والأوزان بنجاح!")
            except Exception as e:
                st.error(f"فشل في تحميل النموذج من الهيكل والأوزان: {str(e)}")
                return None, None
        elif model is None:
            st.error("لم يتم العثور على ملفات النموذج. يرجى تدريب النموذج أولاً.")
            return None, None
            
        # Load label mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            thamudic_letters = mapping_data['thamudic_letters']
            
        return model, thamudic_letters
        
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """
    معالجة الصورة للتعرف
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize
        image = image.resize((128, 128))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="نظام التعرف على النصوص الثمودية",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("نظام التعرف على النصوص الثمودية")
    st.markdown("""
    <style>
        .stTitle {
            text-align: center;
            direction: rtl;
        }
        .stMarkdown {
            direction: rtl;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load model and mapping
    model, thamudic_letters = load_model_and_mapping()
    
    if model is None or thamudic_letters is None:
        return
        
    # File uploader
    st.markdown("<h3 style='text-align: right;'>تحميل صورة للتعرف عليها</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("اختر صورة...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="الصورة المحملة", use_column_width=True)
            
            # Preprocess image
            processed_image = preprocess_image(image)
            if processed_image is None:
                return
                
            # Make prediction
            predictions = model.predict(processed_image)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            # Display results
            st.markdown("<h3 style='text-align: right;'>نتائج التعرف:</h3>", unsafe_allow_html=True)
            
            for idx in top_indices:
                confidence = predictions[idx] * 100
                letter = next((letter for letter in thamudic_letters if letter['index'] == idx), None)
                if letter:
                    st.markdown(
                        f"<div style='text-align: right; direction: rtl;'>"
                        f"<h4>{letter['name']} ({letter['symbol']})</h4>"
                        f"<p>نسبة الثقة: {confidence:.2f}%</p>"
                        f"<p>{letter['description']}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
            logging.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
