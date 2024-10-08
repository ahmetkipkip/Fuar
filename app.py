import streamlit as st
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import cv2
import numpy as np
from PIL import Image
#import mediapipe as mp
from rembg import remove, new_session
from utils import get_ice_servers

session = new_session() # we can't use same time with mediapipe. choose one of them


#mp_selfie_segmentation = mp.solutions.selfie_segmentation
#selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)



#background_image = cv2.imread('r1.jpeg')

st.title("Fuar demo")

def virtual_bg_callback(frame: av.VideoFrame) -> av.VideoFrame:
    print("processing")
    img = frame.to_ndarray(format="bgr24")
    
    # Resize background image to match the frame size
    background_image_resized = cv2.resize(background_image, (img.shape[1], img.shape[0]))

    # Convert the frame to RGB format (MediaPipe expects RGB)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe to get the segmentation mask
    results = selfie_segmentation.process(frame_rgb)

    # The segmentation mask is 2D (height x width), we need to expand it to (height x width x 3)
    mask = results.segmentation_mask > 0.5
    mask_3d = np.dstack((mask, mask, mask))  # Stack the mask along the third dimension

    # Replace the background with the background image using the mask
    output_frame = np.where(mask_3d, img, background_image_resized)
    print("processed")
    
    return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

def rembg_callback(frame: av.VideoFrame) -> av.VideoFrame:
    print("processing")
    img = frame.to_image()
    output_frame = remove(img, session=session)
    
    
    
    
    print("processed")
    
    return av.VideoFrame.from_image(output_frame)
    
webrtc_streamer(
    key="opencv-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=rembg_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
