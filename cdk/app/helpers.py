# helpers.py
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
from datetime import datetime
from matplotlib import cm, colors

try:
    import Image, ImageDraw, ImageFont
except ImportError:
    from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Helper classes
# -----------------------------------------------------------
class Rekognition:
    def __init__(self):
        self.client = boto3.client(
            'rekognition',
            region_name = 'eu-west-2', # not needed
            )

    def predict_labels(self, image_bytes, max_labels=10, min_conf=90):
        response = self.client.detect_labels(
            Image = {'Bytes': image_bytes},
            MaxLabels = max_labels,
            MinConfidence = min_conf,
            )
        return response['Labels']
    
    def return_mask_img(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        imgWidth, imgHeight = image.size
        blank = Image.new('RGB', image.size, (255, 255, 255))
        draw = ImageDraw.Draw(blank)
        response = self.predict_labels(image_bytes)
        
        for idx, label in enumerate(response):
            name = label['Name']
            instances = label['Instances']

            if len(instances) == 0: continue
            for instance in instances:
                confidence = instance['Confidence']
                box = instance['BoundingBox']
                left = imgWidth * box['Left']
                top = imgHeight * box['Top']
                width = imgWidth * box['Width']
                height = imgHeight * box['Height']

                points = (
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left , top + height),
                    (left, top),
                )

                # draw bounding box
                draw.rectangle([left, top, left + width, top + height], fill='black')
                
        return blank


    def draw_labels_on_img(self, image_bytes, return_txt=True, **kwargs):
        image = Image.open(io.BytesIO(image_bytes))
        imgWidth, imgHeight = image.size
        draw = ImageDraw.Draw(image) 
        response = self.predict_labels(image_bytes, **kwargs)
        # font = ImageFont.truetype('IBMPlexSans-Regular.ttf', size=15)
        font = ImageFont.load_default()
        cmap = cm.get_cmap('Paired')
        
        for idx, label in enumerate(response):
            name = label['Name']
            instances = label['Instances']
            color = colors.rgb2hex(cmap(idx)[:3])

            if len(instances) == 0: continue
            for instance in instances:
                confidence = instance['Confidence']
                box = instance['BoundingBox']
                left = imgWidth * box['Left']
                top = imgHeight * box['Top']
                width = imgWidth * box['Width']
                height = imgHeight * box['Height']

                points = (
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left , top + height),
                    (left, top),
                )

                # draw bounding box
                draw.rectangle([left, top, left + width, top + height], outline=color, width=2)

                # draw text
                text = f'{name} {int(confidence)}%'
                w, h = font.getsize(text)
                draw.rectangle([left, top, left + w, top + h], fill='black')
                draw.text(points[0], text, font=font, fill='white')

        if return_txt:
            txt = ['{} - {:.2f}%'.format(label['Name'],label['Confidence']) for label in response]
            return image, txt
        else:
            return image
        
        
class InPainting:
    def __init__(self):
        self.rekognition = Rekognition() 
        self.multiple = 6
        self.INPUT_SIZE = 512  # input image size for Generator
        self.ATTENTION_SIZE = 32 # size of contextual attention
        
    def PIL_to_cv2(self, pil_img):
        np_img = np.array(pil_img.convert('RGB'))
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    
    def PIL_to_image_bytes(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def cv2_to_PIL(self, cv2_im):
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_im)
                
    def run_main(self, input_image, max_size = (1024,1024)):
        with tf.Graph().as_default():
            with open('src/model_weights.pb', "rb") as f:
                output_graph_def = tf.compat.v1.GraphDef()
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")

            with tf.compat.v1.Session() as sess:
                init = tf.compat.v1.global_variables_initializer()
                sess.run(init)
                image_ph = sess.graph.get_tensor_by_name('img:0')
                mask_ph = sess.graph.get_tensor_by_name('mask:0')
                inpainted_512_node = sess.graph.get_tensor_by_name('inpainted:0')
                attention_node = sess.graph.get_tensor_by_name('attention:0')
                mask_512_node = sess.graph.get_tensor_by_name('mask_processed:0')
        
                input_image.thumbnail(max_size)
                image_bytes = self.PIL_to_image_bytes(input_image)
                raw_mask = self.PIL_to_cv2(self.rekognition.return_mask_img(image_bytes))
                raw_img = self.PIL_to_cv2(input_image)
                inpainted = self.inpaint(
                            raw_img, raw_mask, sess, inpainted_512_node, 
                            attention_node, mask_512_node, image_ph, mask_ph, self.multiple)
                return self.cv2_to_PIL(inpainted)
    
    def sort(self, str_lst):
        return [s for s in sorted(str_lst)]

    # reconstruct residual from patches
    def reconstruct_residual_from_patches(self, residual, multiple):
        residual = np.reshape(residual, [self.ATTENTION_SIZE, self.ATTENTION_SIZE, multiple, multiple, 3])
        residual = np.transpose(residual, [0,2,1,3,4])
        return np.reshape(residual, [self.ATTENTION_SIZE * multiple, self.ATTENTION_SIZE * multiple, 3])

    # extract image patches
    def extract_image_patches(self, img, multiple):
        h, w, c = img.shape
        img = np.reshape(img, [h//multiple, multiple, w//multiple, multiple, c])
        img = np.transpose(img, [0,2,1,3,4])
        return img

    # residual aggregation module
    def residual_aggregate(self, residual, attention, multiple):
        residual = self.extract_image_patches(residual, multiple * self.INPUT_SIZE//self.ATTENTION_SIZE)
        residual = np.reshape(residual, [1, residual.shape[0] * residual.shape[1], -1])
        residual = np.matmul(attention, residual)
        residual = self.reconstruct_residual_from_patches(residual, multiple * self.INPUT_SIZE//self.ATTENTION_SIZE)
        return residual

    # resize image by averaging neighbors
    def resize_ave(self, img, multiple):
        img = img.astype(np.float32)
        img_patches = self.extract_image_patches(img, multiple)
        img = np.mean(img_patches, axis=(2,3))
        return img

    # pre-processing module
    def pre_process(self, raw_img, raw_mask, multiple):
        raw_mask = raw_mask.astype(np.float32) / 255.
        raw_img = raw_img.astype(np.float32)

        # resize raw image & mask to desinated size
        large_img = cv2.resize(raw_img,  (multiple * self.INPUT_SIZE, multiple * self.INPUT_SIZE), interpolation = cv2. INTER_LINEAR)
        large_mask = cv2.resize(raw_mask, (multiple * self.INPUT_SIZE, multiple * self.INPUT_SIZE), interpolation = cv2.INTER_NEAREST)

        # down-sample large image & mask to 512x512
        small_img = self.resize_ave(large_img, multiple)
        small_mask = cv2.resize(raw_mask, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation = cv2.INTER_NEAREST)

        # set hole region to 1. and backgroun to 0.
        small_mask = 1. - small_mask
        return large_img, large_mask, small_img, small_mask


    # post-processing module
    def post_process(self, raw_img, large_img, large_mask, res_512, img_512, mask_512, attention, multiple):
        # compute the raw residual map
        h, w, c = raw_img.shape
        low_base = cv2.resize(res_512.astype(np.float32), (self.INPUT_SIZE * multiple, self.INPUT_SIZE * multiple), interpolation = cv2.INTER_LINEAR)
        low_large = cv2.resize(img_512.astype(np.float32), (self.INPUT_SIZE * multiple, self.INPUT_SIZE * multiple), interpolation = cv2.INTER_LINEAR)
        residual = (large_img - low_large) * large_mask

        # reconstruct residual map using residual aggregation module
        residual = self.residual_aggregate(residual, attention, multiple)

        # compute large inpainted result
        res_large = low_base + residual
        res_large = np.clip(res_large, 0., 255.)

        # resize large inpainted result to raw size
        res_raw = cv2.resize(res_large, (w, h), interpolation = cv2.INTER_LINEAR)

        # paste the hole region to the original raw image
        mask = cv2.resize(mask_512.astype(np.float32), (w, h), interpolation = cv2.INTER_LINEAR)
        mask = np.expand_dims(mask, axis=2)
        res_raw = res_raw * mask + raw_img * (1. - mask)

        return res_raw.astype(np.uint8)


    def inpaint(self, 
                raw_img, 
                raw_mask, 
                sess, 
                inpainted_512_node, 
                attention_node, 
                mask_512_node, 
                img_512_ph, 
                mask_512_ph, 
                multiple):

        # pre-processing
        img_large, mask_large, img_512, mask_512 =self.pre_process(raw_img, raw_mask, multiple)

        # neural network
        inpainted_512, attention, mask_512  = sess.run([inpainted_512_node, attention_node, mask_512_node], feed_dict={img_512_ph: [img_512] , mask_512_ph:[mask_512[:,:,0:1]]})

        # post-processing
        res_raw_size = self.post_process(raw_img, img_large, mask_large, \
                     inpainted_512[0], img_512, mask_512[0], attention[0], multiple)

        return res_raw_size
