import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Fruits & Vegetables Recognition System 🍎🥕",
    layout="wide",
    page_icon="🍎"
)

st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        overflow: hidden;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf, #2e7bcf);
        color: white;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 1;
    }
    .stButton>button {
        color: white;
        background-color: #2e7bcf;
    }
    .stButton>button:hover {
        background-color: #1c5c99;
    }
    .header {
        color: #2e7bcf;
        text-align: center;
    }
    .success-msg {
        color: green;
        font-size: 18px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Use Streamlit's HTML component to embed a video as background
st.components.v1.html("""
    <video autoplay muted loop style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: -1;">
        <source src="video1.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
""", height=0, width=0)

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_model.h5")
    return model

model = load_model()

def model_prediction(test_image):
    image = Image.open(test_image).convert('RGB')
    image = image.resize((64, 64))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Dashboard 🌟")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

if app_mode == "Home":
    st.markdown("<h1 class='header'>FRUITS & VEGETABLES RECOGNITION SYSTEM</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("istockphoto-1409236261-612x612.jpg", use_column_width=True)
    with col2:
        st.markdown("""
            <div style='text-align: left;'>
                <h3>Welcome!</h3>
                <p>Upload an image of a fruit or vegetable, and our model will recognize it for you.</p>
                <p>Use the sidebar to navigate through the app.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: left; margin-top: 20px;'>
                <p>
                    Our <b>Fruits & Vegetables Recognition System</b> leverages advanced machine learning algorithms to accurately identify various fruits and vegetables from images. Whether you're a farmer, a retailer, or simply curious about different produce, this tool provides quick and reliable predictions to assist you in your daily tasks.
                </p>
            </div>
            """, unsafe_allow_html=True)
elif app_mode == "Prediction":
    st.markdown("<h1 class='header'>Model Prediction 🧠</h1>", unsafe_allow_html=True)
    st.write("### Upload an image of a fruit or vegetable, or take a photo using your camera, and click **Predict** to see the result.")

    # Add camera input option
    camera_image = st.camera_input("Take a photo:")

    # Add file uploader option
    test_image = st.file_uploader("Or upload an image:", type=['jpg', 'jpeg', 'png'])

    # Use the camera image if available, otherwise use the uploaded image
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Captured Image', use_column_width=True)
        test_image = camera_image  # Use the camera image for prediction
    elif test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if test_image is not None:
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                result_index = model_prediction(test_image)
                with open("labels.txt") as f:
                    content = f.readlines()
                label = [line.strip() for line in content]
                if result_index < len(label):
                    prediction = label[result_index]
                else:
                    prediction = "Unknown"
            st.success(f"The model predicts this as: **{prediction}** 🍎🥕")
    else:
        st.info("Please upload an image or take a photo to get started.")
