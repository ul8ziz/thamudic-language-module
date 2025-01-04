import streamlit as st
import numpy as np
import cv2
import json
from pathlib import Path
from data_preprocessing import ThamudicPreprocessor
from model import ThamudicTranslator

class ThamudicApp:
    def __init__(self):
        # Load character mapping
        with open('data/letters/letter_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create mapping from index to Arabic character
        self.char_mapping = {}
        for letter in data['thamudic_letters']:
            self.char_mapping[letter['index']] = letter['name']
        
        print(f"Loaded {len(self.char_mapping)} characters")
        
        # Initialize preprocessor and model
        self.preprocessor = ThamudicPreprocessor()
        self.translator = ThamudicTranslator(
            model_path='models/thamudic_model.pth',
            char_mapping=self.char_mapping
        )

    def process_image(self, image: np.ndarray) -> tuple[str, list[tuple[str, float]], np.ndarray]:
        """Process an uploaded image and return the translated text, character details, and annotated image."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess the image
            processed_image = self.preprocessor.preprocess_image(image)
            
            # Make a copy of the original image for drawing
            annotated_image = image.copy()
            
            # Segment characters
            characters, bounding_boxes = self.preprocessor.segment_characters(processed_image)
            
            if not characters:
                st.warning("لم يتم العثور على أي حروف في الصورة")
                return "لم يتم العثور على أي حروف في الصورة", [], annotated_image
            
            # Translate each character
            translated_text = ""
            char_details = []
            
            # Draw bounding boxes and predictions
            for i, (char_image, bbox) in enumerate(zip(characters, bounding_boxes)):
                char, confidence = self.translator.predict(char_image)
                
                # Add character details
                if confidence > 0.5:
                    translated_text += char
                    char_details.append((char, confidence))
                    # Draw green box for confident predictions
                    color = (0, 255, 0)  # Green
                else:
                    translated_text += "?"
                    char_details.append(("?", confidence))
                    # Draw red box for low confidence predictions
                    color = (255, 0, 0)  # Red
                
                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                
                # Add text above the box
                text = f"{i+1}: {char} ({confidence:.2%})"
                # Position text based on available space
                if y > 20:  # If there's space above
                    text_y = y - 10
                else:  # Put text inside the box
                    text_y = y + 20
                cv2.putText(annotated_image, text, (x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return translated_text, char_details, annotated_image
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
            return "", [], image

def main():
    st.title("مترجم النصوص الثمودية")
    st.write("قم بتحميل صورة تحتوي على نص ثمودي لترجمته إلى العربية")
    
    # Initialize app
    app = ThamudicApp()
    
    # File uploader
    uploaded_file = st.file_uploader("اختر صورة", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Convert uploaded file to image array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process image and display result
        with st.spinner('جاري معالجة الصورة...'):
            arabic_text, char_details, annotated_image = app.process_image(image_array)
            
            # Display original and annotated images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("الصورة الأصلية")
                st.image(image_array, use_column_width=True)
            with col2:
                st.subheader("الصورة مع تحديد الأحرف")
                st.image(annotated_image, use_column_width=True)
            
            if arabic_text and arabic_text != "لم يتم العثور على أي حروف في الصورة":
                st.success('تمت الترجمة بنجاح!')
                st.write(f"النص المترجم: {arabic_text}")
                
                # Display character details in a table
                st.subheader("تفاصيل الأحرف المكتشفة:")
                char_data = []
                for i, (char, conf) in enumerate(char_details, 1):
                    char_data.append({
                        "رقم الحرف": i,
                        "الحرف": char,
                        "درجة الثقة": f"{conf:.2%}"
                    })
                st.table(char_data)

if __name__ == '__main__':
    main()