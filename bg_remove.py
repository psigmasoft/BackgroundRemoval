import streamlit as st
import numpy as np
from rembg import remove
import cv2
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Remove background from your image")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image(img):
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Cartoonify the image
def cartoonify_image_cv2(img):
    # Ensure the image is a numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Ensure the image is in the correct format
    if len(img.shape) == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # Image with alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    img_color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.adaptiveThreshold(img_gray, 255, 
                                     cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 9)
    cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edge)
    return cartoon

def process_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    img_no_bg = remove(image)
    col2.write("Background removed :wrench:")
    col2.image(img_no_bg)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(img_no_bg), "img_no_bg.png", "image/png")

    img_cartoon = cartoonify_image_cv2(img_no_bg)
    col3.write("Cartoonified Image :art:")
    col3.image(img_cartoon)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download cartoonified image", convert_image(img_cartoon), "img_cartoon.png", "image/png")

col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        process_image(upload=my_upload)
else:
    process_image("./zebra.jpg")
