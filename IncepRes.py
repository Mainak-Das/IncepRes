# Modules
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import fitz
import io
import time
import json
import base64
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from IPython.display import Image as im, display
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

st.set_page_config(
    page_title="IncepRes",
    page_icon="imgs/IncepRes_Favicon.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/IncepRes/IncepRes-2025",
        "About": """
            ## Upload a report to classify cancer with precision
            **GitHub**: https://github.com/IncepRes/IncepRes-2025.git
        """
    }
)

# CDN's for Icons
people_icon_url = "https://cdn-icons-png.flaticon.com/512/456/456212.png"
linkedin_icon_url = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
github_icon_url = "https://cdn-icons-png.flaticon.com/512/733/733609.png"

# ------------------ Main Content Started Here ------------------ #
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

    # span.st-emotion-cache-9ycgxx.e1blfcsg3::after {
    #     width: 1000px;
    #     content: "Drag & Drop Cancer Report Here";
    #     visibility: visible;
    #     position: absolute;
    #     left: 0;
    #     top: 0;
    #     color: #20b8cd;
    #     font-size: 23px;
    #     font-weight: 360;
    #     font-family: "Poppins", sans-serif;
    # }

    div.css-8u98yl.exg6vvm0 > section.css-1iwfxcu.exg6vvm15 {
        border: 4px dashed #20b8cd;
        border-radius: 20px;
        padding: 20px;
        height: 130px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #0e1117; /* Dark background */
        transition: all 0.3s ease-in-out; /* Smooth hover effect */
        color: white; /* Ensuring text is visible */
    }

    /* Optional: Change border color when hovering */
    div.css-8u98yl.exg6vvm0 > section.css-1iwfxcu.exg6vvm15:hover {
        border-color: #1de2ff; /* Slightly brighter cyan */
        background-color: #121826; /* Slightly lighter dark shade */
    }

    /* Improve text styling inside the uploader */
    div.css-8u98yl.exg6vvm0 > section.css-1iwfxcu.exg6vvm15 p {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff; /* White text for contrast */
    }

    span.css-9ycgxx.exg6vvm12,
    .css-1fttcpj.exg6vvm11 > span.css-9ycgxx.exg6vvm12,
    .css-1fttcpj.exg6vvm11 > span:nth-child(1),
    .css-1fttcpj > span.css-9ycgxx,
    .exg6vvm12 {
        visibility: hidden;
        position: relative;
    }

    span.css-9ycgxx.exg6vvm12::after,
    .css-1fttcpj.exg6vvm11 > span.css-9ycgxx.exg6vvm12::after,
    .css-1fttcpj.exg6vvm11 > span:nth-child(1)::after,
    .css-1fttcpj > span.css-9ycgxx::after,
    .exg6vvm12::after {
        content: "Drag & Drop Cancer Report Here";
        color: #20b8cd;
        font-size: 23px;
        font-weight: 360;
        font-family: "Poppins", sans-serif;
        visibility: visible;
        position: absolute;
        left: 0;
        top: -7px;
        width: 120%; /* Increase the width of the container */
        text-align: center;
        white-space: nowrap; /* Prevents text from wrapping */
    }

    .css-1fttcpj.exg6vvm11 > small.css-7oyrr6.euu6i2w0 {
        margin-top: -10px; /* Adds space above the element */
        /* Or you can use transform for a smoother effect */
        transform: translateY(10px); /* Moves the text down smoothly */
    }

    button.css-1ak6eu7.edgvbvh10 {
        font-family: "Poppins", sans-serif;
        font-size: 18px;
        font-weight: 450;
        color: #20b8cd;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
# ------------------ END of CSS for File Uploader ------------------ #

# ------------------ File Uploader ------------------ #
hide_file_name_and_remove_button = """
    <style>
        div.stFileUploaderFile.st-emotion-cache-12xsiil.e1blfcsg10 {
            display: flex !important;
            align-items: center; 
        }
        
        div.stFileUploaderFile.st-emotion-cache-12xsiil.e1blfcsg10 button {
            margin-left: 10px;
        }

        .success_toast {
            position: fixed;
            top: 9%;
            left: 50%;
            transform: translateX(-50%);
            background-image: linear-gradient(135deg, rgba(76, 175, 80, 0.8), rgba(139, 195, 74, 0.8));
            color: #0e1117;
            padding: 12px 10px;
            border-radius: 12px;
            font-size: 17px;
            font-weight: 600;
            font-family: "Poppins", sans-serif;
            text-align: center;
            z-index: 1000;
            opacity: 0.5;
            animation: fadeInOut 4s ease-in-out forwards;
        }

        .warning_toast {
            position: fixed;
            top: 9%;
            left: 50%;
            transform: translateX(-50%);
            background-image: linear-gradient(135deg, rgba(255, 187, 51, 1), rgba(255, 140, 0, 1)); /* Light yellowish-orange */
            color: #0e1117;
            padding: 12px 10px;
            border-radius: 12px;
            font-size: 17px;
            font-weight: 600;
            font-family: "Poppins", sans-serif;
            text-align: center;
            z-index: 1000;
            opacity: 1;  /* Full opacity */
            animation: fadeInOut 4s ease-in-out forwards;
        }

        @keyframes fadeInOut {
            0% { opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { opacity: 0; }
        }
    </style>
"""
st.markdown(hide_file_name_and_remove_button, unsafe_allow_html=True)
image = None

# Function to display custom toast message
def show_success_toast(message):
    st.markdown(f'<div class="success_toast">{message}</div>', unsafe_allow_html=True)

def show_warning_toast(message):
    st.markdown(f'<div class="warning_toast">{message}</div>', unsafe_allow_html=True)

# File uploader widget
input_file = st.file_uploader("", type=["png", "jpg", "jpeg", "pdf"])

# Check if a file is already uploaded
if input_file is not None:
    file_type = input_file.type
    file_name = input_file.name

    # if st.session_state['uploaded_file'] is None:
    #     st.session_state['uploaded_file'] = file_name
    if file_type in ["image/png", "image/jpeg"]:
        image = Image.open(input_file).convert("RGB")
        # print(image)
        show_success_toast(f"Report uploaded successfully...")
    elif file_type == "application/pdf":
        pdf_document = fitz.open(stream=input_file.read(), filetype="pdf")
        show_success_toast(f"PDF uploaded successfully...")
    else:
        st.error("Unsupported file format!")

# ----------------------------- SELECT BOX & SUBMIT BUTTON -------------------------------- #

class_names = ['all_benign', 'all_early', 'all_pre', 'all_pro', 'brain_glioma', 'brain_menin', 'brain_tumor', 'breast_benign', 'breast_malignant', 'cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi', 'colon_aca', 'colon_bnt', 'kidney_normal', 'kidney_tumor', 'lung_aca', 'lung_bnt', 'lung_scc', 'lymph_cll', 'lymph_fl', 'lymph_mcl', 'oral_normal', 'oral_scc']

class_names_df = {
    0 : ['all_benign', 'Acute Lymphoblastic Leukemia - Benign', 'Non-cancerous, healthy cells', 'Healthy'],
    1 : ['all_early', 'Acute Lymphoblastic Leukemia - Early', 'Early stages of leukemia', 'Cancerous'],
    2 : ['all_pre', 'Acute Lymphoblastic Leukemia - Prestage', 'Pre-stage abnormal cells', 'Cancerous'],
    3 : ['all_pro', 'Acute Lymphoblastic Leukemia - Advanced', 'Advanced leukemia cells', 'Cancerous'],
    4 : ['brain_glioma', 'Glioma (Brain)', 'Most common brain tumor', 'Cancerous'],
    5 : ['brain_menin', 'Meningioma (Brain)', 'Tumors affecting brain membranes', 'Cancerous'],
    6 : ['brain_tumor', 'Pituitary Tumor (Brain)', 'Tumors affecting the pituitary gland', 'Cancerous'],
    7 : ['breast_benign', 'Benign Breast Tissue', 'Non-cancerous breast tissues', 'Healthy'],
    8 : ['breast_malignant', 'Malignant Breast Tissue', 'Cancerous breast tissues', 'Cancerous'],
    9 : ['cervix_dyk', 'Cervical Cancer - Dyskeratotic', 'Abnormal cell growth', 'Cancerous'],
    10: ['cervix_koc', 'Cervical Cancer - Koilocytotic', 'Cells showing changes from viral infections (e.g., HPV)', 'Cancerous'],
    11: ['cervix_mep', 'Cervical Cancer - Metaplastic', 'Cells changed from one type to another (precancerous)', 'Cancerous'],
    12: ['cervix_pab', 'Cervical Cancer - Parabasal', 'Immature squamous cells', 'Cancerous'],
    13: ['cervix_sfi', 'Cervical Cancer - Superficial-Intermediate', 'More mature squamous cells', 'Cancerous'],
    14: ['colon_aca', 'Colon Adenocarcinoma', 'Cancerous cells of the colon', 'Cancerous'],
    15: ['colon_bnt', 'Colon Benign Tissue', 'Healthy colon tissues', 'Healthy'],
    16: ['kidney_normal', 'Normal Kidney Tissue', 'Healthy kidney tissues', 'Healthy'],
    17: ['kidney_tumor', 'Tumor-affected Kidney Tissue', 'Tumor-affected kidney tissues', 'Cancerous'],
    18: ['lung_aca', 'Lung Adenocarcinoma', 'Cancerous cells of the lung', 'Cancerous'],
    19: ['lung_bnt', 'Benign Lung Tissue', 'Healthy lung tissues', 'Healthy'],
    20: ['lung_scc', 'Lung Squamous Cell Carcinoma', 'Aggressive lung cancer type', 'Cancerous'],
    21: ['lymph_cll', 'Chronic Lymphocytic Leukemia', 'Slow-progressing blood cancer', 'Cancerous'],
    22: ['lymph_fl', 'Follicular Lymphoma', 'Slow-growing non-Hodgkin lymphoma', 'Cancerous'],
    23: ['lymph_mcl', 'Mantle Cell Lymphoma', 'Aggressive form of lymphoma', 'Cancerous'],
    24: ['oral_normal', 'Normal Oral Cells', 'Healthy oral tissues', 'Healthy'],
    25: ['oral_scc', 'Oral Squamous Cell Carcinoma', 'Cancerous oral cells', 'Cancerous']
}

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
    
def display_gradcam(image_array, heatmap, alpha=0.4):
    # Ensure image is in the correct format (float32 for scaling)
    img = np.array(image_array, dtype=np.float32)

    # Rescale heatmap to range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]
    
    # Get RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))  # Match original image size
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)  # Ensure valid pixel values

    # # Display Grad-CAM
    # plt.figure(figsize=(6, 6))
    # plt.imshow(superimposed_img)
    # plt.axis("off")  # Hide axes
    # plt.show()
    
    return superimposed_img

def show_imgwithheat(img, heatmap, alpha=0.4, return_array=False):
    """Show the image with heatmap.

    Args:
        img_path: string.
        heatmap: image array, get it by calling grad_cam().
        alpha: float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)
    try:
        display(imgwithheat)
    except NameError:
        imgwithheat.show()

    if return_array:
        return superimposed_img

def grad_cam_plus(model, img,
                  layer_name="block5_conv3", label_name=None,
                  category_id=None):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    # img_tensor = np.expand_dims(img, axis=0)
    img_tensor = img
    conv_layer = model.get_layer(layer_name)
    heatmap_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                # if label_name is not None:
                #     print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap

def lime_explainer(image_array, model):
    explainer = LimeImageExplainer()
    # print("Image shape:", image_array[0].shape)
    # Explain the instance with LIME
    explanation = explainer.explain_instance(image_array[0], model.predict, top_labels=5, hide_color=None, num_samples=100)
    # explanation = explainer.explain_instance(image_array[0], model.predict, top_labels=5, hide_color=0, num_samples=512)

    # Get explanation for top label
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    
    temp_resized = cv2.resize(temp, (512, 512), interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)

    return temp_resized, mask_resized

def classify_operations(image, model):
    image_res = image.resize((72,72))
    # Convert to NumPy array
    image_array = np.array(image_res)
    # print(image_array.shape)
    image_array = np.expand_dims(image_array, axis=0)
    
    grad_image_array = np.array(image)
    pred_val_prob = model.predict(image_array)
    # print(pred_val_prob)
    pred_val = np.argmax(pred_val_prob, axis=-1)[0]
    pred_class = class_names_df[pred_val][1]
    state = class_names_df[pred_val][3]
    pred_val_prob = pred_val_prob[0, pred_val]
    
    return pred_class, pred_val, image_array, grad_image_array, pred_val_prob, state

def gradcam_operations(image):
    model = keras.models.load_model('incep-res__26class_2dec2024.h5')
    pred_class, pred_val, image_array, grad_image_array, pred_val_prob, state = classify_operations(image, model)
    
    heatmap = make_gradcam_heatmap(image_array, model, 'add_8')        
    super = display_gradcam(grad_image_array, heatmap)
    
    return pred_class, pred_val, super, pred_val_prob, heatmap, state

def gradcampp_operations(image):
    model = keras.models.load_model('incep-res__26class_2dec2024.h5')
    pred_class, pred_val, image_array, grad_image_array, pred_val_prob, state = classify_operations(image, model)
    
    heatmap_plus = grad_cam_plus(model, image_array, layer_name='add_8')       
    super = show_imgwithheat(grad_image_array, heatmap_plus, return_array=True)
    
    return pred_class, pred_val, super, pred_val_prob, heatmap_plus, state
    
def lime_operations(image):
    model = keras.models.load_model('incep-res__26class_2dec2024.h5')
    pred_class, pred_val, image_array, grad_image_array, pred_val_prob = classify_operations(image, model)
    
    temp, mask = lime_explainer(image_array, model)       
    # super = show_imgwithheat(grad_image_array, heatmap_plus, return_array=True)
    
    return grad_image_array, pred_class, pred_val, mask, pred_val_prob

# Functions for Full Screen Spinner
def show_full_screen_spinner():
    st.markdown(
        """
        <style>
        .full-screen-loader {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: #0e1117;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 999999;
        }

        .loader {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            max-width: 6rem;
            margin-top: 3rem;
            margin-bottom: 3rem;
        }

        .loader:before,
        .loader:after {
            content: "";
            position: absolute;
            border-radius: 50%;
            animation: pulsOut 1.8s ease-in-out infinite;
            filter: drop-shadow(0 0 1rem rgba(32, 184, 205, 0.75)); /* Updated shadow color */
        }

        .loader:before {
            width: 100%;
            padding-bottom: 100%;
            box-shadow: inset 0 0 0 1rem #20b8cd; /* Updated color */
            animation-name: pulsIn;
        }

        .loader:after {
            width: calc(100% - 2rem);
            padding-bottom: calc(100% - 2rem);
            box-shadow: 0 0 0 0 #20b8cd; /* Updated color */
        }

        @keyframes pulsIn {
            0% {
                box-shadow: inset 0 0 0 1rem #20b8cd; /* Updated color */
                opacity: 1;
            }
            50%, 100% {
                box-shadow: inset 0 0 0 0 #20b8cd; /* Updated color */
                opacity: 0;
            }
        }

        @keyframes pulsOut {
            0%, 50% {
                box-shadow: 0 0 0 0 #20b8cd; /* Updated color */
                opacity: 0;
            }
            100% {
                box-shadow: 0 0 0 1rem #20b8cd; /* Updated color */
                opacity: 1;
            }
        }
        </style>

        <div class="full-screen-loader">
            <div class="loader"></div>
        </div>

        """, unsafe_allow_html=True
    )

# Function to hide the full-screen spinner (overlay)
def hide_full_screen_spinner():
    st.markdown(
        """
        <style>
        .full-screen-loader {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True
    )

# ----------------------------------- New Classify Button with a Spinner Added on 02/04/2025 -----------------------------------
col1, col2 = st.columns([5, 1])
with col1:
    options = ["Select Output Type..."] + ["Grad-CAM", "Grad-CAM++"]
    output_type = st.selectbox(
        label="",
        options=options,
        index=0,
        key="output_type"
    )

with col2:
    classify_button = st.button("Classify ➤", key="submit_btn")

if classify_button:
    # Check for missing inputs or parameters
    if output_type == "Select Output Type..." and ('input_file' not in locals() or input_file is None):
        show_warning_toast(f"Please upload a report and select an output parameter!")
    elif output_type == "Select Output Type...":
        show_warning_toast(f"Please select an output parameter!")
    elif 'input_file' not in locals() or input_file is None:
        show_warning_toast(f"Please upload a report!")
    else:
        # This block runs when the input and output are valid
        
        # Show the full-screen spinner while performing the analysis operations
        show_full_screen_spinner()

        with st.spinner():
            # ---------- Grad-Cam Analysis ----------
            if output_type == "Grad-CAM":
                # Perform Grad-CAM analysis
                pred_class, pred_val, super, pred_val_prob, heatmap, state = gradcam_operations(image)
                # Process the heatmap and apply colormap
                if heatmap.max() > 1:
                    heatmap = heatmap.astype(np.float32) / 255.0  # Normalize to [0,1]
                colormap = plt.cm.plasma(heatmap)
                colormap = (colormap[:, :, :3] * 255).astype(np.uint8)

                # Determine border color based on probability
                probability = pred_val_prob * 100

                # # Swastik's color scheme
                if state == 'Cancerous':
                    border_color = "#a4161a"  # Reddish
                # elif 51 <= probability <= 80:
                #     border_color = "#ffa200"  # Yellowish
                else:
                    border_color = "#09a129"  # Greenish
                # # Swastik's color scheme
                
                # # Mainak's Initial color scheme
                # if probability <= 50:
                #     border_color = "#a4161a"  # Reddish
                # elif 51 <= probability <= 80:
                #     border_color = "#ffa200"  # Yellowish
                # else:
                #     border_color = "#09a129"  # Greenish
                # # Mainak's Initial color scheme

                # Custom CSS for styling the Report Analysis section with dynamic border
                st.markdown(
                    f"""
                    <style>
                        @keyframes glowing-border {{
                            0% {{ box-shadow: 0px 0px 5px {border_color}; }}
                            50% {{ box-shadow: 0px 0px 12px {border_color}; }}
                            100% {{ box-shadow: 0px 0px 5px {border_color}; }}
                        }}
                        .report-container {{
                            border: 2px solid {border_color};
                            border-radius: 15px;
                            padding: 15px 30px;
                            background-color: #121826;
                            animation: glowing-border 1.5s infinite alternate;
                            margin-bottom: 30px;
                            font-family: 'Poppins', sans-serif;
                            text-align: left;
                            box-shadow: 0px 0px 10px rgba(32, 184, 205, 0.5);
                        }}
                        .report-title {{
                            color: #20b8cd;
                            font-size: 28px;
                            font-weight: bold;
                            text-align: center;
                            padding-bottom: 15px;
                        }}
                        .report-text-container {{
                            display: flex;
                            flex-direction: column;
                            justify-content: left;
                            gap: 10px;
                            font-size: 22px;
                            font-weight: 600;
                            color: #20b8cd;
                            padding-left: 15px;
                        }}
                        .report-text-container span.pred-value {{
                            font-size: 22px;
                            font-weight: 400;
                            color: #ffffff;
                            justify-content: left;
                        }}
                        .image-subheader {{
                            text-align: center;
                            font-size: 24px;
                            font-weight: 600;
                            color: #20b8cd;
                            margin-top: -5px;
                        }}
                        .stImage {{
                            margin-bottom: 5px;
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # First row: Report Analysis in a customized container
                with st.container():
                    st.markdown(
                        f"""
                        <div class="report-container">
                            <div class="report-title">Report Analysis</div>
                            <div class="report-text-container">
                                <span>Cancer Type: <span class="pred-value">{pred_class}</span></span>
                                <span>Prediction Strength: <span class="pred-value">{probability:.2f} %</span></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Second row: Input Image, Grad-CAM Image & Heatmap side-by-side
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(input_file, use_column_width=True)
                    st.markdown('<div class="image-subheader">Input Image</div>', unsafe_allow_html=True)
                with col2:
                    st.image(super, use_column_width=True)
                    st.markdown('<div class="image-subheader">Grad-CAM Image</div>', unsafe_allow_html=True)
                with col3:
                    st.image(colormap, use_column_width=True)
                    st.markdown('<div class="image-subheader">Heatmap</div>', unsafe_allow_html=True)
            # ---------- End of Grad-Cam Analysis ----------

            # ---------- Grad-Cam++ Analysis ----------
            elif output_type == "Grad-CAM++":
                pred_class, pred_val, super, pred_val_prob, heatmap, state = gradcampp_operations(image)
                
                if heatmap.max() > 1:
                    heatmap = heatmap.astype(np.float32) / 255.0  # Normalize to [0,1]
                colormap = plt.cm.plasma(heatmap)
                colormap = (colormap[:, :, :3] * 255).astype(np.uint8)

                probability = pred_val_prob * 100

                # # Swastik's color scheme
                if state == 'Cancerous':
                    border_color = "#a4161a"  # Reddish
                # elif 51 <= probability <= 80:
                #     border_color = "#ffa200"  # Yellowish
                else:
                    border_color = "#09a129"  # Greenish
                # # Swastik's color scheme
                
                # # Mainak's Initial color scheme
                # if probability <= 50:
                #     border_color = "#a4161a"  # Reddish
                # elif 51 <= probability <= 80:
                #     border_color = "#ffa200"  # Yellowish
                # else:
                #     border_color = "#09a129"  # Greenish
                # # Mainak's Initial color scheme

                # Custom CSS for styling the Report Analysis section with dynamic border
                st.markdown(
                    f"""
                    <style>
                        @keyframes glowing-border {{
                            0% {{ box-shadow: 0px 0px 5px {border_color}; }}
                            50% {{ box-shadow: 0px 0px 12px {border_color}; }}
                            100% {{ box-shadow: 0px 0px 5px {border_color}; }}
                        }}
                        .report-container {{
                            border: 2px solid {border_color};
                            border-radius: 15px;
                            padding: 15px 30px;
                            background-color: #121826;
                            animation: glowing-border 1.5s infinite alternate;
                            margin-bottom: 30px;
                            font-family: 'Poppins', sans-serif;
                            text-align: left;
                            box-shadow: 0px 0px 10px rgba(32, 184, 205, 0.5);
                        }}
                        .report-title {{
                            color: #20b8cd;
                            font-size: 28px;
                            font-weight: bold;
                            text-align: center;
                            padding-bottom: 15px;
                        }}
                        .report-text-container {{
                            display: flex;
                            flex-direction: column;
                            justify-content: left;
                            gap: 10px;
                            font-size: 22px;
                            font-weight: 600;
                            color: #20b8cd;
                            padding-left: 15px;
                        }}
                        .report-text-container span.pred-value {{
                            font-size: 22px;
                            font-weight: 400;
                            color: #ffffff;
                            justify-content: left;
                        }}
                        .image-subheader {{
                            text-align: center;
                            font-size: 24px;
                            font-weight: 600;
                            color: #20b8cd;
                            margin-top: -5px;
                        }}
                        .stImage {{
                            margin-bottom: 5px;
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # First row: Report Analysis in a customized container
                with st.container():
                    st.markdown(
                        f"""
                        <div class="report-container">
                            <div class="report-title">Report Analysis</div>
                            <div class="report-text-container">
                                <span>Cancer Type: <span class="pred-value">{pred_class}</span></span>
                                <span>Prediction Strength: <span class="pred-value">{probability:.2f} %</span></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Second row: Input Image, Grad-CAM Image & Heatmap side-by-side
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(input_file, use_column_width=True)
                    st.markdown('<div class="image-subheader">Input Image</div>', unsafe_allow_html=True)
                with col2:
                    st.image(super, use_column_width=True)
                    st.markdown('<div class="image-subheader">Grad-CAM++ Image</div>', unsafe_allow_html=True)
                with col3:
                    st.image(colormap, use_column_width=True)
                    st.markdown('<div class="image-subheader">Heatmap</div>', unsafe_allow_html=True)
            # ---------- End of Grad-Cam++ Analysis ----------
            
            time.sleep(3)
        hide_full_screen_spinner()
# ----------------------------------- End of Classify Button with a Spinner Added on 02/04/2025 -----------------------------------

st.markdown("""
    <style>
        div.stHorizontalBlock.st-emotion-cache-ocqkz7.e6rk8up0 {
            margin: 0px !important;
            padding: 0px !important;
        }
            
        div[data-baseweb="select"] span[title="Select Output Type..."] {
            color: gray !important;
            font-family: "Poppins", sans-serif;
            margin-top: 0;
        }

        div.row-widget.stButton > button.css-1ak6eu7.edgvbvh10 {
            margin-top: 3.5px;  
            # transform: translateY(10px);
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

st.markdown(
    """
    <style>
        h1, h2, h3 {
            color: #20b8cd !important;
            font-weight: 600;
        }

        .sidebar-img {
            display: block;
            # margin-left: auto;
            # margin-right: auto;
            margin: 0 0;
            width: 100%;
        }

        hr {
            border: 1px solid #20b8cd;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and display sidebar image
img_path = "imgs/sidebar_incepres_logo.png"
img_base64 = img_to_base64(img_path)
if img_base64:
    st.sidebar.markdown(
        f'<img class="sidebar-img" src="data:image/png;base64,{img_base64}">',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")

st.sidebar.header("About")
st.markdown(
    """
    <style>
        .st-emotion-cache-1espb9k.egexzqm0 > p {
            text-align: justify;                
        }

        .css-nahz7x.e16nr0p34 > p:only-child {
            text-align: justify;
            font-size: 15px;
        }

        button.css-dw5rzi.edgvbvh3 > svg.e1fb0mya1.css-fblp2m.ex0cdmw0 {
            # font-family: "FontAwesome"; /* If using FontAwesome */
            # content: "\f104"; /* Left arrow in FontAwesome */
            color: blue;
        }
        
        div.e1fqkh3o4{
            padding-top: 50px;     
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("Welcome to IncepRes, an advanced machine learning-based cancer detection model designed to enhance diagnostic accuracy and speed.")
st.sidebar.markdown("Cancer detection remains a critical challenge, with traditional methods often yielding inaccurate results. IncepRes utilizes deep learning architectures, combining the power of Inception and Residual networks, to identify various cancer types from medical images. With a compact size and efficient processing, IncepRes offers faster training times and superior precision compared to traditional models.")

st.sidebar.markdown("---")

st.sidebar.header("Developers")
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
                <a href="https://www.linkedin.com/in/swastik-karmakar-541bb7252" target="_blank">
                    <img src="{linkedin_icon_url}" alt="LinkedIn">
                </a>
                <a href="https://www.github.com/Darth-Hannibal" target="_blank">
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
                <a href="https://github.com/aishikmaitra" target="_blank">
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
                <a href="https://www.linkedin.com/in/ayush-das-499a12247" target="_blank">
                    <img src="{linkedin_icon_url}" alt="LinkedIn">
                </a>
                <a href="https://github.com/infinity-ayush" target="_blank">
                    <img src="{github_icon_url}" alt="GitHub">
                </a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# st.sidebar.markdown("---")
# st.sidebar.header("FAQ's")
# ------------------ End of Side Bar ------------------ #