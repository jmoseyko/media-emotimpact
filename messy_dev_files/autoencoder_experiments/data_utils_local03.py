#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:42:28 2018

@author: chloeloughridge
"""

import cv2
import os

# function for reading video frames into a file
# source: source: https://www.life2coding.com/extract-frame-video-file-using-opencv-python/ 
def extractFrames_toFile(pathIn, pathOut):
    os.mkdir(pathOut)
 
    cap = cv2.VideoCapture(pathIn)
    count = 0
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
 
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# function for grabbing a frame (array of pixel values) directly from video
# returns success value and the frame pixel values
# I don't actually foresee this function being very useful
def get_frm_vid(vid_filePath):
    vidcap = cv2.VideoCapture(vid_filePath)
    success, frame = vidcap.read()
    return success, frame

# function for grabbing a frame that's saved to a folder
def get_frm_folder(img_path):
    img = cv2.imread(img_path)
    return img


# function for reshaping the frame to 64x64 for autoencoder 
def downsize_forAuto(frame):
    resized_img = cv2.resize(frame, (64,64))
    return resized_img
    
    
    
    
    
    
    
    
    
    
    
    