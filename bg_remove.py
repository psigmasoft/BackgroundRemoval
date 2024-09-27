import streamlit as st
import numpy as np
from rembg import remove
import cv2
from PIL import Image
from io import BytesIO
import base64
import requests

##################
# NOTES:
# To deploy for public access: https://docs.streamlit.io/deploy/tutorials
# Free Amazon EC2 instance: https://towardsdatascience.com/how-to-deploy-a-streamlit-app-using-an-amazon-free-ec2-instance-416a41f69dc3?
##################

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Cartoonify your favourite Lioness")
st.write(
    ":dog: Uploading an image using the button on the left. Full quality images can be downloaded from the sidebar. :grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Default value for slider_block
#slider_block = 11

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
def cartoonify_image_cv2_edges(img):
    # Ensure the image is a numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Ensure the image is in the correct format
    if len(img.shape) == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # Image with alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
# cv2.bilateralFilter:

# d: 9
# sigmaColor: 75
# sigmaSpace: 75

# cv2.cvtColor:

# Convert the image to grayscale using cv2.COLOR_BGR2GRAY.
# cv2.medianBlur:

# ksize: 7 (Ensure it's an odd number)
# cv2.adaptiveThreshold:

# maxValue: 255
# adaptiveMethod: cv2.ADAPTIVE_THRESH_MEAN_C
# thresholdType: cv2.THRESH_BINARY
# blockSize: 9 (Ensure it's an odd number greater than 1)
# C: 2

    img_color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    #applying median blur to smoothen an image
    # get the blue value, round to nearest odd number
    blur_value = slider_blur
    if blur_value % 2 == 0:
        blur_value += 1
    img_blur = cv2.medianBlur(img_gray, blur_value)
    
    # Ensuring blockSize is an odd integer greater than 1
    # if slider_block % 2 == 0:
    #     slider_block += 1
    # if slider_block <= 1:
    #     slider_block = 3

        # Ensuring blockSize is an odd integer greater than 1
    # if slider_adaptive % 2 == 0:
    #     slider_adaptive += 1
    # if slider_adaptive <= 1:
    #     slider_adaptive = 3

    # Apply adaptive threshold
    block_size = slider_block_size
    if block_size % 2 == 0:
        block_size += 1
    if block_size <= 1:
        block_size = 3

    #retrieving the edges for cartoon effect

    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2)

    return img_edge

# def cartoonify_image_edges_to_toon(img, img_edge):
#     # Ensure the image is a numpy array
#     if not isinstance(img, np.ndarray):
#         img = np.array(img)
    
#     # Ensure the image is in the correct format
#     if len(img.shape) == 2:  # Grayscale image
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.shape[2] == 4:  # Image with alpha channel
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

#     #applying bilateral filter to remove noise 
#     #and keep edge sharp as required
#     img_filtered = cv2.bilateralFilter(img, d=slider_d, sigmaColor=slider_sigmaColor, sigmaSpace=slider_sigmaSpace)

#     #masking edged image with our "BEAUTIFY" image
#     img_cartoon = cv2.bitwise_and(img_filtered, img_filtered, mask=img_edge)

#     return img_cartoon

def process_image(upload):
    image = Image.open(upload)
    col_orig.write("Original Image :camera:")
    col_orig.image(image)

    img_no_bg = remove(image)
    col2_no_bg.write("Background removed :wrench:")
    col2_no_bg.image(img_no_bg)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(img_no_bg), "img_no_bg.png", "image/png")

    img_edges = cartoonify_image_cv2_edges(img_no_bg)
    col_edges.write("Edges :art:")
    col_edges.image(img_edges)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download edges image", convert_image(img_edges), "img_edges.png", "image/png")

    # img_cartoon = cartoonify_image_edges_to_toon(img_no_bg, img_edges)
    # col_cartoon.write("Cartoonified Image :art:")
    # col_cartoon.image(img_cartoon)
    # st.sidebar.markdown("\n")
    # st.sidebar.download_button("Download cartoonified image", convert_image(img_cartoon), "img_cartoon.png", "image/png")

# Columns for the images
col_orig, col2_no_bg, col_edges, col_cartoon  = st.columns(4)

# Sliders to adjust the cartoonify effect
slider_blur = st.sidebar.slider("Blur", 1, 11, 7)
slider_block_size = st.sidebar.slider("Block Size", 1, 21, 3)

# Display text showing the parameters used under the columns
col_cartoon.write("Parameters used:")
col_cartoon.write(f"Blur: {slider_blur}")
col_cartoon.write(f"Block Size: {slider_block_size}")



# cv2.adaptiveThreshold in edge image conversion
#slider_adaptive = st.sidebar.slider("Adaptive Threshold", 1, 255, 100)
#slider_block = st.sidebar.slider("Block Size", 1, 255, 9)
#slider_c = st.sidebar.slider("C", 1, 255, 9)

# cv2.bilateralFilter sliders
# slider_d = st.sidebar.slider("d", 1, 255, 9)
# slider_sigmaColor = st.sidebar.slider("sigmaColor", 1, 255, 75)
# slider_sigmaSpace = st.sidebar.slider("sigmaSpace", 1, 255, 75)

my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        process_image(upload=my_upload)
else:
    process_image("./zebra.jpg")






# Can I run this on Ainize? Offers free GPU usage?
# https://ainize.ai/psi1104/White-box-Cartoonization?
# https://master-white-box-cartoonization-psi1104.endpoint.ainize.ai/

# Ainize PW: bxfPbrpzkA6bY0
    
# Created account but looks like the free tier is no longer available
def cartoonify_whitebox(img):
# Post to the Ainize API
#  curl -X POST "https://master-white-box-cartoonization-psi1104.endpoint.ainize.ai/predict" \
#      -H "accept: image/jpg" \
#      -H "Content-Type: multipart/form-data" \
#      -F "file_type=image" \
#      -F "source=@your_image.png;type=image/png"

    # issue a CURL request to the Ainize API
    response = requests.post("https://master-white-box-cartoonization-psi1104.endpoint.ainize.ai/predict",
                                headers={"accept": "image/jpg", "Content-Type": "multipart/form-data"},
                                files={"file_type": "image", "source": img})
    return response


