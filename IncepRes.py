# Modules 
# import openai
import streamlit as st
import logging
import numpy as np
from PIL import Image, ImageEnhance
import time
import json
import requests
import base64
# from openai import OpenAI, OpenAIError


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20
API_DOCS_URL = "https://docs.streamlit.io/library/api-reference"

# Retrieve and validate API key
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
# if not OPENAI_API_KEY:
#     st.error("Please add your OpenAI API key to the Streamlit secrets.toml file.")
#     st.stop()

# Assign OpenAI API Key
# openai.api_key = OPENAI_API_KEY
# client = openai.OpenAI()

# Streamlit Page Configuration
st.set_page_config(
    page_title="IncepRes",
    page_icon="imgs/IncepRes_Favicon.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/AdieLaine/Streamly",
        "About": """
            ## Upload a report to classify cancer with precision
            ### Powered using GPT-4o-mini

            **GitHub**: https://github.com/mainak-das

            The AI Assistant named, Streamly, aims to provide the latest updates from Streamlit,
            generate code snippets for Streamlit widgets,
            and answer questions about Streamlit's latest features, issues, and more.
            Streamly has been trained on the latest Streamlit updates and documentation.
        """
    }
)

# Streamlit Title
# st.title("Upload a report to classify cancer with precision")

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
            margin-top: 70px;
            line-height: 1.3;

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Additional styling for another line */
        .line2 {
            margin-top: 10px;
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

    # CSS for highlighting text bg
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

    # Apply the CSS
    st.markdown(highlight_css, unsafe_allow_html=True)

    # Sub-heading with the highlighted "IncepRes" text
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

# ------------------ Image Uploader ------------------ #
# # img_file_buffer = st.file_uploader('Upload an image', type=['png', 'jpg'])
img_file_buffer = st.file_uploader(
    '',
    type=['png', 'jpg'],
    accept_multiple_files=False
    )

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
# ------------------ End of Image Uploader ------------------ #

#Submit Button
# left, middle, right = st.columns(3)
# # if middle.button("Material button", icon=":material/mood:", use_container_width=True):
# if middle.button("Classify", use_container_width=True, type="primary"):
#     middle.markdown("You clicked the Classify button.")

col1, col2, col3 = st.columns([1, 3, 1], vertical_alignment="top")
with col1:
    pass
with col2:
    st.markdown("""
        <style>
            .centered-title {
                font-size: 25px;
                text-align: center;
                margin-top: 20px;
                margin-bottom: -50px;
            }

            .stButton {
                margin-top: 0px;
                margin-bottom: 20px;
                display: flex;
                justify-content: center;
            }
            
            .stButton > button {
                margin: 0 auto;
                width: 200px;
                background-color: #20b8cd;
                border-color: #20b8cd;
                color: #ffffff;
                border-radius: 10px;
                padding: 0.5rem 0.5rem;
                font-size: 2rem; 
                font-weight: 500; 
                transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
                text-align: center;
                text-decoration: none; 
                display: inline-block;
                font-family: 'Oswald', sans-serif;  
            }

            .stButton > button:hover {
                background-color: #1993A4;
                border-color: #1993A4;
                color: #ffffff;
                text-decoration: none;
                transition: transform 0.5s ease, box-shadow 0.5s ease, background-color 0.5s ease;
                transform: scale(1.001);
            }

            .stButton p{
                font-size: 1.3rem;
                font-family: 'Poppins', sans-serif;
                font-weight: 700;
                color: #262730;
            }            
        </style>
    """, unsafe_allow_html=True)

# ------------------- Submit Button -------------------- #
if st.button("Classify"):
    if img_file_buffer is None:
        st.warning("Please upload an Image!")
    else:
        with st.spinner('Analysing...Please wait'):
            time.sleep(3)
            st.write("Button Clicked")
# ------------------- End of Submit Button -------------------- #

# ------------------ Helper Functions ------------------ #
def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def long_running_task(duration):
    """
    Simulates a long-running operation.

    Parameters:
    - duration: int, duration of the task in seconds

    Returns:
    - str: Completion message
    """
    time.sleep(duration)
    return "Long-running operation completed."

@st.cache_data(show_spinner=False)
def load_and_enhance_image(image_path, enhance=False):
    """
    Load and optionally enhance an image.

    Parameters:
    - image_path: str, path of the image
    - enhance: bool, whether to enhance the image or not

    Returns:
    - img: PIL.Image.Image, (enhanced) image
    """
    img = Image.open(image_path)
    if enhance:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
    return img

@st.cache_data(show_spinner=False)
def load_streamlit_updates():
    """Load the latest Streamlit updates from a local JSON file."""
    try:
        with open("data/streamlit_updates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON: {str(e)}")
        return {}

# def get_streamlit_api_code_version():
#     """
#     Get the current Streamlit API code version from the Streamlit API documentation.

#     Returns:
#     - str: The current Streamlit API code version.
#     """
#     try:
#         response = requests.get(API_DOCS_URL)
#         if response.status_code == 200:
#             return "1.36"
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error connecting to the Streamlit API documentation: {str(e)}")
#     return None

# def display_streamlit_updates():
#     """Display the latest updates of the Streamlit."""
#     with st.expander("Streamlit 1.36 Announcement", expanded=False):
#         st.markdown("For more details on this version, check out the [Streamlit Forum post](https://docs.streamlit.io/library/changelog#version).")

# def initialize_conversation():
#     """
#     Initialize the conversation history with system and assistant messages.

#     Returns:
#     - list: Initialized conversation history.
#     """
#     assistant_message = "Hello! I am Streamly. How can I assist you with Streamlit today?"

#     conversation_history = [
#         {"role": "system", "content": "You are Streamly, a specialized AI assistant trained in Streamlit."},
#         {"role": "system", "content": "Streamly, is powered by the OpenAI GPT-4o-mini model, released on July 18, 2024."},
#         {"role": "system", "content": "You are trained up to Streamlit Version 1.36.0, release on June 20, 2024."},
#         {"role": "system", "content": "Refer to conversation history to provide context to your response."},
#         {"role": "system", "content": "You were created by Madie Laine, an OpenAI Researcher."},
#         {"role": "assistant", "content": assistant_message}
#     ]
#     return conversation_history

@st.cache_data(show_spinner=False)
def get_latest_update_from_json(keyword, latest_updates):
    """
    Fetch the latest Streamlit update based on a keyword.

    Parameters:
    - keyword (str): The keyword to search for in the Streamlit updates.
    - latest_updates (dict): The latest Streamlit updates data.

    Returns:
    - str: The latest update related to the keyword, or a message if no update is found.
    """
    for section in ["Highlights", "Notable Changes", "Other Changes"]:
        for sub_key, sub_value in latest_updates.get(section, {}).items():
            for key, value in sub_value.items():
                if keyword.lower() in key.lower() or keyword.lower() in value.lower():
                    return f"Section: {section}\nSub-Category: {sub_key}\n{key}: {value}"
    return "No updates found for the specified keyword."

# def construct_formatted_message(latest_updates):
#     """
#     Construct formatted message for the latest updates.

#     Parameters:
#     - latest_updates (dict): The latest Streamlit updates data.

#     Returns:
#     - str: Formatted update messages.
#     """
#     formatted_message = []
#     highlights = latest_updates.get("Highlights", {})
#     version_info = highlights.get("Version 1.36", {})
#     if version_info:
#         description = version_info.get("Description", "No description available.")
#         formatted_message.append(f"- **Version 1.36**: {description}")

#     for category, updates in latest_updates.items():
#         formatted_message.append(f"**{category}**:")
#         for sub_key, sub_values in updates.items():
#             if sub_key != "Version 1.36":  # Skip the version info as it's already included
#                 description = sub_values.get("Description", "No description available.")
#                 documentation = sub_values.get("Documentation", "No documentation available.")
#                 formatted_message.append(f"- **{sub_key}**: {description}")
#                 formatted_message.append(f"  - **Documentation**: {documentation}")
#     return "\n".join(formatted_message)

@st.cache_data(show_spinner=False)
def on_chat_submit(chat_input, latest_updates):
    """
    Handle chat input submissions and interact with the OpenAI API.

    Parameters:
    - chat_input (str): The chat input from the user.
    - latest_updates (dict): The latest Streamlit updates fetched from a JSON file or API.

    Returns:
    - None: Updates the chat history in Streamlit's session state.
    """
    user_input = chat_input.strip().lower()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = initialize_conversation()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        model_engine = "gpt-4o-mini"
        assistant_reply = ""

        if "latest updates" in user_input:
            assistant_reply = "Here are the latest highlights from Streamlit:\n"
            highlights = latest_updates.get("Highlights", {})
            if highlights:
                for version, info in highlights.items():
                    description = info.get("Description", "No description available.")
                    assistant_reply += f"- **{version}**: {description}\n"
            else:
                assistant_reply = "No highlights found."
        else:
            response = client.chat.completions.create(
                model=model_engine,
                messages=st.session_state.conversation_history
            )
            assistant_reply = response.choices[0].message.content

        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except OpenAIError as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"OpenAI Error: {str(e)}")

# def initialize_session_state():
#     """Initialize session state variables."""
#     if "history" not in st.session_state:
#         st.session_state.history = []
#     if 'conversation_history' not in st.session_state:
#         st.session_state.conversation_history = []

# def main():
#     """
#     Display Streamlit updates and handle the chat interface.
#     """
#     initialize_session_state()

#     if not st.session_state.history:
#         initial_bot_message = "Hello! How can I assist you with Streamlit today?"
#         st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
#         st.session_state.conversation_history = initialize_conversation()

# ------------------ End of Helper Functions ------------------ #

# ------------------ Side Bar ------------------ #
    # Load and display sidebar image

    # st.markdown(
    #     """
    #     <style>
    #         svg.e1fexwmo1.st-emotion-cache-qsoh6x.ex0cdmw0 path:nth-child(2) {
    #             fill: red;
    #         }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

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
            # f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            f'<img src="data:image/png;base64,{img_base64}">',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")

    # Add instructions on how to use the app to the sidebar
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
    # Add the sidebar header
    # st.sidebar.header("How to use IncepRes")
    # st.sidebar.markdown("---")

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

if __name__ == "__main__":
    main()