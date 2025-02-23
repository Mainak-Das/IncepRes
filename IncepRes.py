# Modules
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import time
import json
import base64

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20
API_DOCS_URL = "https://docs.streamlit.io/library/api-reference"

st.set_page_config(
    page_title="IncepRes",
    page_icon="imgs/IncepRes_Favicon.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/mainak-das/IncepRes",
        "About": """
            ## Upload a report to classify cancer with precision
            ### Powered using GPT-4o-mini
            **GitHub**: https://github.com/mainak-das/IncepRes
        """
    }
)

col1 = st.container()
with col1:
    st.markdown("""
    <style>
        .gradient-background {
            background: linear-gradient(142deg,#20b8cd,#5acdd6,#7fdde3,#1a9eb5,#157f99);
            background-size: 300% 300%;
            animation: gradient-animation 5s ease infinite;
        }

        @keyframes gradient-animation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        .full-width {
            width: 100%;
            text-align: center;
            font-size: 58px;
            font-family: "Poppins", sans-serif;
            font-weight: 1000;
            color: #212529;
            line-height: 1.3;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="full-width gradient-background">Hybrid & Lightweight AI<br><span class="line2">Delivering Faster Cancer Classification</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <style>
    .sub-heading {
        width: 100%;
        text-align: center;
        font-size: 20px;
        font-family: 'Montserrat', sans-serif;
        font-weight: 200;
        color: #f4f4f4;
        margin-top: -15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # CSS for highlighting text BG
    highlight_css = """
    <style>
    .container {
        line-height: 1.4;
        text-align: center;
        padding: 44px;
        color: #333;
        font-family: 'Montserrat', sans-serif;
    }
    h1 {
        font-size: 50px;
    }
    p {
        font-size: 18px;
    }
    p small {
        font-size: 80%;
        color: #666;
    }

    .highlight-container, .highlight {
        position: relative;
        font-weight: bold;
        font-family: 'Montserrat', sans-serif;
    }

    .highlight-container {
        display: inline-block;
        font-family: 'Montserrat', sans-serif;
    }

    .highlight-container:before {
        content: " ";
        display: block;
        height: 80%;
        width: 106%;
        margin-left: -3.5px;
        margin-right: -3px;
        position: absolute;
        background: #1aa0b2;
        transform: rotate(-1deg);
        top: 4px;
        left: 1px;
        border-radius: 50% 20% 50% 20%;
        padding: 10px 3px 3px 10px;
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    """
    st.markdown(highlight_css, unsafe_allow_html=True)

    # CSS for the Tagline
    st.markdown(
    '''
    <style>
        .highlight-container {
            margin-right: 7px;
        }
    </style>
    <div class="sub-heading">
        <span class="highlight-container">
            <span class="highlight">IncepRes</span>
        </span>
        Where Speed Meets Precision in Cancer Detection
    </div>
    ''',
    unsafe_allow_html=True
)

# ------------------ CSS for Image Uploader ------------------ #
st.markdown(
    """ 
    <style>
    div.block-container {
        margin-top: 10px;
    }

    .stFileUploader > section {
        font-size: 22px;
        font-family: "Poppins", sans-serif;    
        margin-top: 0px !important;
        height: 200px;
        padding: 10px 40px;
        position: relative;
        border-radius: 20px;
        border: 4px solid transparent;
        background:
            linear-gradient(#262730, #262730) padding-box,
            repeating-linear-gradient(
                45deg,
                #20b8cd,
                #20b8cd 10px,
                transparent 10px,
                transparent 15px
            ) border-box;   
    }

    div[data-baseweb="select"] > div {
        min-height: 50px;
        display: flex;
        align-items: center;
        font-size: 20px !important; 
        padding-left: 7px;
        font-family: "Poppins", sans-serif;   
        border: .1px solid #20b8cd;
    }

    span.st-emotion-cache-9ycgxx.e1blfcsg3 {
        visibility: hidden;
        position: relative;
    }

    span.st-emotion-cache-9ycgxx.e1blfcsg3::after {
        width: 1000px;
        content: "Drag & Drop Cancer Report Here";
        visibility: visible;
        position: absolute;
        left: 0;
        top: 0;
        color: #20b8cd;
        font-size: 23px;
        font-weight: 360;
        font-family: "Poppins", sans-serif;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

img_file_buffer = st.file_uploader(
    '',
    type=['pdf', 'png', 'jpg'],
    accept_multiple_files=False
)

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
# ----------------------- End of CSS for Image Uploader ----------------------- #

# ----------------------------- SELECT BOX & SUBMIT BUTTON -------------------------------- #
col1, col2 = st.columns([5, 1])

with col1:
    options = ["Select Output Type..."] + ["Grad-CAM", "Grad-CAM++", "LIME"]
    output_type = st.selectbox(
        label="",
        options=options,
        index=0,
        key="output_type"
    )

with col2:
    classify_button = st.button("Classify ➤", key="submit_btn")
if classify_button:
    if output_type == "Select Output Type..." and ('img_file_buffer' not in locals() or img_file_buffer is None):
        st.warning("⚠  Please upload an image and select an output parameter!")
    elif output_type == "Select Output Type...":
        st.warning("⚠  Please select an output parameter!")
    elif 'img_file_buffer' not in locals() or img_file_buffer is None:
        st.warning("⚠  Please upload an image!")
    else:
        with st.spinner('Analysing... Please wait'):
            time.sleep(3)
        st.write("✅ Button Clicked!")

st.markdown("""
    <style>
        div[data-baseweb="select"] span[title="Select Output Type..."] {
            color: gray !important;
            font-family: "Poppins", sans-serif;
            margin-top: 0;
        }

        .stButton {
            margin-top: 28px;
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
        }
        
        .stButton > button {
            width: 200px;
            height: 48px;
            background: linear-gradient(142deg,#20b8cd,#5acdd6,#7fdde3,#1a9eb5,#157f99);
            background-size: 300% 300%;
            animation: gradient-animation 5s ease infinite;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 0.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            text-align: center;
            text-decoration: none; 
            display: inline-block;
            font-family: "Poppins", sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: white;
            cursor: pointer;
        }

        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }

        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stButton > button p {
            font-family: "Poppins", sans-serif !important;
            font-size: 20px !important;
            font-weight: 700;
            color: #0e1117 !important;
        }
    </style>
    """, unsafe_allow_html=True)
# ------------------- End of Selector & Button ------------------- #

# ----------------------- Helper Functions ----------------------- #
def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        # logging.error(f"Error converting image to base64: {str(e)}")
        return None
# -------------------- End of Helper Functions -------------------- #

# ---------------------------- Side Bar ---------------------------- #
st.markdown(
    """
    <style>
        svg.e1fexwmo1.st-emotion-cache-qsoh6x.ex0cdmw0 path:nth-child(2) {
            color: #20b8cd;
        }
    </style>
    """,
    unsafe_allow_html=True
)

img_path = "imgs/sidebar_incepres_logo.png"
img_base64 = img_to_base64(img_path)
if img_base64:
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}">',
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")

st.markdown(
    """
    <style>
        .stVerticalBlock .stHeading h2 {
            color: #20b8cd;
            font-weight: 600;
            font-size: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("About")
st.markdown(
    """
    <style>
        .st-emotion-cache-1espb9k.egexzqm0 > p {
            text-align: justify;                
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("Welcome to IncepRes, an advanced machine learning-based cancer detection model designed to enhance diagnostic accuracy and speed.")
st.sidebar.markdown("Cancer detection remains a critical challenge, with traditional methods often yielding inaccurate results. IncepRes utilizes deep learning architectures, combining the power of Inception and Residual networks, to identify various cancer types from medical images. With a compact size and efficient processing, IncepRes offers faster training times and superior precision compared to traditional models.")

st.sidebar.markdown("---")

st.sidebar.header("Developers")

# Person Icons & Developer Names
icon_url = "https://cdn-icons-png.flaticon.com/512/456/456212.png"
st.sidebar.markdown(
    f"""
    <style>
        .sidebar-container {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .sidebar-container img {{
            width: 18px;
            height: 20px;
            margin-right: 8px;
            padding-bottom: 1px;
            filter: invert(1);
        }}
        .sidebar-container span {{
            color: white;
            font-size: 18px;
        }}
    </style>
    <div class="sidebar-container">
        <img src="{icon_url}" alt="Person Icon">
        <span>Swastik Karmakar</span>
    </div>
    <div class="sidebar-container">
        <img src="{icon_url}" alt="Person Icon">
        <span>Mainak Das</span>
    </div>
    <div class="sidebar-container">
        <img src="{icon_url}" alt="Person Icon">
        <span>Aishik Maitra</span>
    </div>
    <div class="sidebar-container">
        <img src="{icon_url}" alt="Person Icon">
        <span>Ayush Das</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.header("FAQ's")
# ------------------ End of Side Bar ------------------ #