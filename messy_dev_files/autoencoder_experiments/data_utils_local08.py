#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:42:28 2018

@author: chloeloughridge
"""

# This file is for dealing with data from the LIRIS ACCEDE dataset


import cv2
import os
import numpy as np

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
    
  
# function for creating one hot vector out of fear annotations
    #STILL NEED TO TEST
def fear_oneHot(movie_length, fear_path):
    # load start and end times from file
    y_data = np.loadtxt(fear_path, skiprows=1)
    y_data_input = np.zeros((movie_length))
    
    # need to address corner case if y_data is 1D array
    if type(y_data[0]) == np.float64:
        y_data_ext = np.zeros([1,2])
        y_data_ext[:,:] = np.asarray(y_data)
    else:
        y_data_ext = y_data 
        
    
# =============================================================================
#     try:
#         test = y_data.shape[0]
#         y_data_ext = y_data
#         for i in range(test):
#             # access the start time number and end time number
#             start = int(y_data_ext[i][0])
#             end = int(y_data_ext[i][1])
#             # set the elements between these indices in the zeros array to one
#             y_data_input[start] = 1 #maybe superfluous
#             y_data_input[end] = 1
#             y_data_input[start:end] = 1
#     except:
#         y_data_ext = []
#         y_data_ext.append(y_data)
#         for i in range(y_data.shape[0]):
#             # access the start time number and end time number
#             start = int(y_data_ext[i][0])
#             end = int(y_data_ext[i][1])
#             # set the elements between these indices in the zeros array to one
#             y_data_input[start] = 1 #maybe superfluous
#             y_data_input[end] = 1
#             y_data_input[start:end] = 1
# =============================================================================
        
    
    # for each element in first dimension of the y_data array
    for i in range(y_data_ext.shape[0]):
        # access the start time number and end time number
        start = int(y_data_ext[i][0])
        end = int(y_data_ext[i][1])
        # set the elements between these indices in the zeros array to one
        y_data_input[start] = 1 #maybe superfluous
        y_data_input[end] = 1
        y_data_input[start:end] = 1
        
    return y_data_input

# a function for finding the correct fc6 folder
def get_fc6_directory(movie_num):
    return os.path.join("visual_features_part01",
                       "{}".format(movie_num))

# a function for iterating through all of the files in a folder and loading them into input_data
def load_Xinput(directory):
    X_input = np.zeros([212, 4096]) # MAGIC NUMBERS
    count = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            input_data = np.loadtxt(os.path.join(directory, file), delimiter=',')
            X_input[count, :] = np.asarray(input_data)[:]
            #print(os.path.join(directory, filename))
            count = count + 1
            continue
        else:
            continue
    return X_input
    
    
    
    
    
    
    
    