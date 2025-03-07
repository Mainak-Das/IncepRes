# Modules
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import fitz
import io
import time
import json
import base64

st.set_page_config(
    page_title="IncepRes",
    page_icon="imgs/IncepRes_Favicon.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/mainak-das/IncepRes",
        "About": """
            ## Upload a report to classify cancer with precision
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
            font-size: 50px;
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

    # CSS for the Tagline (IncepRes Where Speed Meets Precision in Cancer Detection)
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

# ------------------ CSS for File Uploader ------------------ #
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
        height: 140px;
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
# ------------------ END of CSS for File Uploader ------------------ #

# ------------------ File Uploader ------------------ #
input_file = st.file_uploader("", type=["png", "jpg", "jpeg", "pdf"])

if input_file is not None:
    file_type = input_file.type

    if file_type in ["image/png", "image/jpeg"]:
        image = Image.open(input_file)

    elif file_type == "application/pdf":
        pdf_document = fitz.open(stream=input_file.read(), filetype="pdf")

    else:
        st.error("Unsupported file format!")
# ------------------ End of File Uploader ------------------ #

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
    if output_type == "Select Output Type..." and ('input_file' not in locals() or input_file is None):
        st.warning("⚠  Please upload an image and select an output parameter!")
    elif output_type == "Select Output Type...":
        st.warning("⚠  Please select an output parameter!")
    elif 'input_file' not in locals() or input_file is None:
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
# Developer Names
people_icon_url = "https://cdn-icons-png.flaticon.com/512/456/456212.png"
linkedin_icon_url = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
github_icon_url = "https://cdn-icons-png.flaticon.com/512/733/733609.png"

st.sidebar.markdown(
    f"""
    <style>
        .sidebar-container {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            position: relative;
        }}

        .sidebar-container img.person-icon {{
            width: 16px;
            height: 18px;
            margin-right: 6px;
            filter: invert(1);
        }}

        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}

        .tooltip span {{
            font-size: 20px;
            font-weight: 500;
        }}

        .tooltip .tooltiptext {{
            width: 70px;
            visibility: hidden;
            background-color: #20b8cd;
            color: white;
            text-align: center;
            padding: 6px 10px;
            border-radius: 7px;
            position: absolute;
            left: 105%;
            top: -3px;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }}

        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}

        .tooltiptext img {{
            height: 19px;
            width: auto;
            filter: invert(1) brightness(1000%);
        }}
    </style>

    <div class="sidebar-container">
        <img src="{people_icon_url}" class="person-icon" alt="Person Icon">
        <div class="tooltip">
            <span>Swastik Karmakar</span>
            <div class="tooltiptext">
                <a href="https://www.linkedin.com/in/swastik-karmakar-541bb7252/" target="_blank">
                    <img src="{linkedin_icon_url}" alt="LinkedIn">
                </a>
                <a href="https://www.linkedin.com/in/swastik-karmakar-541bb7252/" target="_blank">
                    <img src="{github_icon_url}" alt="GitHub">
                </a>
            </div>
        </div>
    </div>

    <div class="sidebar-container">
        <img src="{people_icon_url}" class="person-icon" alt="Person Icon">
        <div class="tooltip">
            <span>Mainak Das</span>
            <div class="tooltiptext">
                <a href="https://www.linkedin.com/in/mainakdas2001/" target="_blank">
                    <img src="{linkedin_icon_url}" alt="LinkedIn">
                </a>
                <a href="https://github.com/mainak-das" target="_blank">
                    <img src="{github_icon_url}" alt="GitHub">
                </a>
            </div>
        </div>
    </div>

    <div class="sidebar-container">
        <img src="{people_icon_url}" class="person-icon" alt="Person Icon">
        <div class="tooltip">
            <span>Aishik Maitra</span>
            <div class="tooltiptext">
                <a href="https://www.linkedin.com/in/aishik-maitra-4199b5250/" target="_blank">
                    <img src="{linkedin_icon_url}" alt="LinkedIn">
                </a>
                <a href="https://github.com/aishik-maitra" target="_blank">
                    <img src="{github_icon_url}" alt="GitHub">
                </a>
            </div>
        </div>
    </div>

    <div class="sidebar-container">
        <img src="{people_icon_url}" class="person-icon" alt="Person Icon">
        <div class="tooltip">
            <span>Ayush Das</span>
            <div class="tooltiptext">
                <a href="https://www.linkedin.com/in/ayush-das-499a12247/" target="_blank">
                    <img src="{linkedin_icon_url}" alt="LinkedIn">
                </a>
                <a href="https://github.com/ayush-das" target="_blank">
                    <img src="{github_icon_url}" alt="GitHub">
                </a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.header("FAQ's")
# ------------------ End of Side Bar ------------------ #