import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Cache the TFLite interpreter loading
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=r'D:\iot project\mobilenetv1_quant.tflite')
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {str(e)}")
        return None

# Load the model
interpreter = load_tflite_model()
if interpreter is None:
    st.stop()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Camel', 'Cow', 'Goat']

def preprocess_img(uploaded_file):
    img = Image.open(uploaded_file).resize((96, 96)).convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array.astype(np.float32)

# Streamlit UI
st.title("Animal Classifier (TFLite)")
uploaded_file = st.file_uploader("Upload animal image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Your Image", use_container_width=True)

    with col2:
        with st.spinner('Analyzing image...'):
            try:
                processed_img = preprocess_img(uploaded_file)

                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], processed_img)

                # Run inference
                interpreter.invoke()

                # Get output tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                # Dequantize if model is quantized
                if output_details[0]['dtype'] == np.uint8:
                    scale, zero_point = output_details[0]['quantization']
                    output_data = scale * (output_data.astype(np.float32) - zero_point)

                # Interpret results
                predicted_class = class_names[np.argmax(output_data)]
                confidence = np.max(output_data) * 100

                st.success("## Prediction Results")
                st.metric("Animal", predicted_class)
                st.metric("Confidence", f"{confidence:.1f}%")

                st.write("### Probability Distribution")
                prob_data = {class_names[i]: float(output_data[i]) for i in range(len(class_names))}
                st.bar_chart(prob_data)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
