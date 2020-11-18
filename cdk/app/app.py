# app.py
#
# Web App Demo
# -----------------------------------------------------------
# Author: Mariano Nicolas Metallo
# -----------------------------------------------------------

# Import libraries
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
import logging
import io
import os
import time
import json
import base64
import yaml
import boto3
import botocore
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from pathlib import Path

try:
    import Image, ImageDraw, ImageFont
except ImportError:
    from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Init class
# -----------------------------------------------------------
from helpers import InPainting
magic_trick = InPainting()

# Helper Function
# -----------------------------------------------------------
def show_images(input_img, output_img):
    f = plt.figure(figsize=(20,20))
    f.add_subplot(1,2,1)
    plt.imshow(input_img)
    f.add_subplot(1,2,2)
    plt.imshow(output_img)
    plt.show(block=True)
    st.pyplot(bbox_inches='tight')

# Main Function
# -----------------------------------------------------------
def main():
    st.markdown('''
    # Legendary Streamlit Demo
    ## What are we building?

    We are going to create a web app that can do a magic trick for us. 
    That trick will be to make certain things "disappear" from images. 
    For that, we are going to use something that's called **Image Inpainting**, 
    or the process where you take the missing part of an image and you restore 
    it based on the background information. We are going to look into one of 
    the most recent research papers on the topic that will soon be published 
    on CVPR 2020 called **"Contextual Residual Aggregation for Ultra 
    High-Resolution Image Inpainting"**. You can see here the paper by 
    [Zili Yi et al. (2020)](https://arxiv.org/abs/2005.09704) and its code 
    implementation in 
    [this repo](https://github.com/Atlas200dk/sample-imageinpainting-HiFill).
    
    But before we actually run any image inpainting, we need to generate a 
    mask image that will only show black pixels onto a white foreground where 
    there is something that we want to make disappear from the photo. You can 
    do this manually or you can let a computer do it for you. If you choose 
    the second option, then your best bet is to use a "Semantic Segmentation" 
    algorithm that will do a pixel-wise classification that most closely 
    resembles the shape of that object. But, as we want to keep this as 
    simple as possible, we are going to run "Object Detection" instead, 
    drawing bounding boxes on top of these objects. The result won't be as 
    good, but it will be good enough. We are going to use 
    [AWS Rekognition](https://aws.amazon.com/rekognition/) for its simplicity 
    of use and inference time. You can always learn more about these 
    two (and more) computer vision applications from the 
    [GluonCV website](https://gluon-cv.mxnet.io/contents.html) or some 
    other [very good frameworks](https://github.com/facebookresearch/detectron2).

    The following is the result that we expect to get. You can see the input image 
    (see below for author information) on the left and the output image on the 
    right. You can find all the steps that you need to get that result in the 
    `demo.ipynb` notebook.
    ''')
    st.image(
        'src/magic_trick.png',
        use_column_width=True)

    # read input image
    st.write('---')
    st.header('Read image')
    st.image(
        'src/input_img.png',
        caption='Illustration by https://blush.design/artists/vijay-verma',
        use_column_width=True,
    )
    options = st.radio('Please choose any of the following options',
        (
            'Choose example from library',
            'Download image from URL',
            'Upload your own image',
        )
    )

    input_image = None
    if options == 'Choose example from library':
        image_files = list(sorted([x for x in Path('test_images').rglob('*.jpg')]))
        selected_file = st.selectbox(
            'Select an image file from the list', image_files
        )
        st.write(f'You have selected `{selected_file}`')
        input_image = Image.open(selected_file)
    elif options == 'Download image from URL':
        image_url = st.text_input('Image URL')
        try:
            r = requests.get(image_url)
            input_image = Image.open(io.BytesIO(r.content))
        except Exception:
            st.error('There was an error downloading the image. Please check the URL again.')
    elif options == 'Upload your own image':
        uploaded_file = st.file_uploader("Choose file to upload")
        if uploaded_file:
            input_image = Image.open(io.BytesIO(uploaded_file.decode()))
            st.success('Image was successfully uploaded')

    if input_image:
        st.image(input_image, use_column_width=True)
        st.info('''
        Image will be resized to fit within `(1024,1024)`
        pixels for easier processing.
        ''')
    else:
        st.warning('There is no image loaded.')

    # do a magic trick!
    st.header('Run prediction')
    st.write('')
    if input_image and st.button('Do a magic trick!'):
        try:
            with st.spinner():
                output_image = magic_trick.run_main(input_image)
                show_images(input_image, output_image)
        except Exception as e:
            st.error(e)
            st.error('There was an error processing the input image')
    if not input_image: st.warning('There is no image loaded')

# Run Application
# -----------------------------------------------------------
if __name__ == '__main__':
    main()