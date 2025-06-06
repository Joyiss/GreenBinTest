import keras
import cv2
import numpy as np
import streamlit as st
import google.generativeai as genai
import time

api_key = st.secrets["gemini_api_key"]

genai.configure(api_key=api_key)

gen_model = genai.GenerativeModel("gemini-1.5-flash")

class_names = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]


@st.cache_resource
def load_model():
    return keras.models.load_model('model/trashClassifier.keras')

def predict(file_bytes):
    model = load_model()
    img_np = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    prediction = model.predict(np.array([img]) / 255.0)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index]) * 100
    return class_names[index], confidence

def generate_response(prediction, confidence):
    prompt = """Please give a brief explanation whether this object is recyclable or compostable or should be put in trash. If the word is too broad,
    generalize it to the most common items. 
    - metal: aluminium cans, steel cans
    - biological: food scrapes, leaves, fruits, rotten vegetables, molded bread
    - trash: dirty diapers, face masks, and tooth brushes
    Do not ask the user to check with their local recycle centers for more  specific information 
    as I have already provided that warning. Also if you consider an object to be recyclable, trash, or compostable, do not say that the object is not the other two classes.
    Just say if it is trash, recyclable, or compost. Include additional information if necessary. Also, include a little fun fact about the object. If the confidence is below
    90 percent, inform the user that information may be inaccurate. Also add emojis where appropriate. However, dont add emojis everywhere.
    Here is the object: """ + prediction + """Here is confidence score: """ + str(confidence)
    response = gen_model.generate_content(prompt)
    return response

def stream_response(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.08)

st.set_page_config("Green Bin", "Images/icon.png", layout="wide")
st.logo("Images/logo.png", size="large", icon_image="Images/icon.png")
st.image("Images/logo.png", width=200)

tab1, tab2, tab3 = st.tabs([":material/home: Home", ":material/developer_guide: How to Use", ":material/info: About"])

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #e5e5f7;
    opacity: 0.8;
    background-image: radial-gradient(#444cf7 0.5px, #ffffff 0.5px);
    background-size: 10px 10px;
}
[data-testid="stHeader"] {
    background-color: #e5e5f7;
    opacity: 0.1;
    background-image: radial-gradient(#444cf7 0.5px, #e5e5f7 0.5px);
    background-size: 10px 10px;
}
"""

st.markdown(page_bg_img,unsafe_allow_html=True)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Please select a file", type="jpg")
        st.divider()
        enable = st.toggle("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)
        predict_button = st.button("Analyze :brain:", use_container_width=True)

    with col2:
        if predict_button:
            if uploaded_file and picture:
                st.warning("Please only provide one image")
            elif uploaded_file:
                st.image(uploaded_file)
                with st.spinner("Sorting Trash..."):
                    model_prediction, confidence = predict(uploaded_file)
                    gen_model_text = generate_response(model_prediction, confidence)
                    st.write(f"Confidence: {confidence:.2f}%")
                    st.write_stream(stream_response(gen_model_text.text))
                st.balloons()
            elif picture:
                st.image(picture)
                with st.spinner("Sorting Trash..."):
                    model_prediction, confidence = predict(picture)
                    gen_model_text = generate_response(model_prediction, confidence)
                    st.write(f"Confidence: {confidence:.2f}%")
                    st.write_stream(stream_response(gen_model_text.text))
                st.balloons()
            else:
                st.warning("Please provide an image")

with tab2:
    tab2_col1, tab2_col2 = st.columns((1,1))

    with tab2_col1:
        st.header("How to Use")
        st.write("""
        1. Upload or take a photo of your waste item.  
        2. The model predicts the material class.  
        3. An AI tells you if it's recyclable, compostable, or trash.
        """)

    with tab2_col2:
        st.write("")

    st.info("""**Important Note:** Our model provides general recycling, composting, and trash recommendations based on common guidelines. 
    However, recycling rules can vary by city or region. For the most accurate and up-to-date information, 
    please check with your local waste management, recycling, or sanitation center.""")



with tab3:
    st.header("About")
    st.markdown("Smart waste disposal powered by AI.")

    with st.expander("**Why it is important**"):
        st.write("""
        At Green Bin, we believe that sustainable habits start with simple actions. 
        Our mission is to help people make informed, environmentally conscious decisions 
        about how they dispose of everyday items.
        """)

    with st.expander("**Who It's For**"):
        st.markdown("""
        - Anyone unsure about where their waste goes  
        """)

    with st.expander("**What Makes Green Bin Different**"):
        st.write("""
        Most recycling apps rely on static databases. Green Bin uses real-time object detection 
        and generative AI to give personalized guidance on trash, compost, or recycling—just from a photo.
        """)

    with st.expander("**Technology Behind the App**"):
        st.write("""
        - **MobileNetV2** for image classification (92% accuracy)
        - **Gemini** for context-aware recycling instructions  
        - Built with Python and Streamlit for fast, interactive deployment  
        """)

    with st.expander("**Data Source & License**"):
        st.write("- Contains information from [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification), which is made available here under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).")

    st.divider()

    st.markdown("**Get in Touch**")
    st.markdown("Questions, feedback, or collaboration? Contact us at m.rajashekarreddyus@gmail.com")
    st.link_button("Check out my website", url="https://rakshanmallela.streamlit.app/")
