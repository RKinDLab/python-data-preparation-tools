# Script to extract RGB images from ROSBAGS
# Author: Azmyin Md. Kamal
# Date : 01/25/2023
# Version: 1.0

"""
Master script that creates the dataset similar to EuRoC dataset created by ASL dataset

Requirements
** ROSBAGS directory must be in the home directory folder
** > Python 3.6 needed

Useage
--Directory of rosbags, name of bag file, name of dataset directory, camera topic name
-- "dataset_dir" argument should be passed in the following format "LSU_iCORE/experiment_name i.e LSU_iCORE/SR1
-- Put the camera_calib.yaml file manually once the dataset is created

Changelog
* 01/25/23 - Initial version

"""

# Import modules
import numpy as np # Numerical Python library 
import sys
import rospy # Pure Python client library for ROS
import os # Misc. operating system interface functions
import time # Python timing functions
import argparse # To accept user arguments from commandline
import cv2 # OpenCV
import tqdm # Prints a progress bar on console
from pathlib import Path # To find the "home" directory location
import shutil # High level folder operation tool
#print(cv2.__version__) # Recommended 4.2.0.34

# Import libraries to read ros messages
from sensor_msgs.msg import Image
from std_msgs.msg import String
import rosbag # ROS library to manipulate rosbags 
from cv_bridge import CvBridge, CvBridgeError # Library to convert image messages to numpy array 


# ***************************** User Defined Functions *************************************

# ------------------------------------------------------------------------------------------
def debug_lock():
    # CORE FUNCTION
    # Locks system in an infinite loop when called
    # Debugging function
    #print(f"LOCK")
    print(f"LOCK")
    while (1):
        pass
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def current_milli_time():
    # HELPER FUNCTION
    # Function that returns a time tick in milliseconds
    return round(time.time() * 1000)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def show_image(img, win_name, win_size, win_move_loc,wait_key):
    # HELPER FUCTION
    # Display this image

    """
    Parameters
    img --> image, numpy array
    win_name --> window name, string
    win_pos --> size of window of window, list in form [width,height], list
    win_move_loc --> position of window, list in form [X_pixel_cord, Y_pixel_cord], list
    """

    try:
        cv2.imshow(win_name,img)
        cv2.waitKey(wait_key) # Minimum delay needed??
        if win_size is not None:
            cv2.resizeWindow(win_name, win_size[0],win_size[1]) # Resize window 
        if win_move_loc is not None:
            cv2.moveWindow(win_name,win_move_loc[0],win_move_loc[1]) # Move window
    except:
        print("Error in displaying image")
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
def convert_from_ros_to_opencv_imgz(imgz_msg, br):
    # CORE FUNCTION
    # Using Cvbridge, convert ros image message to opencv numpy array at bgr color format
    try:
        # Convert ROS image to a openCV image
        cv_image = None
        cv_image = br.imgmsg_to_cv2(imgz_msg, desired_encoding='bgr8') # `bgr8` needed by Pytorch
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    return cv_image
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def find_nearest(array, value):
    # HELPER FUNCTION
    # Returns list index and value from list which is closest match to the target value
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array/2566508#2566508
    # Define working variables
    val = None
    idx = None
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    val = array[idx]
    return idx, val
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def create_folder_once(folder_name_with_path):
    """
    Parameters
    folder_name_with_path
    """
    # CORE FUNCTION
    # Creates a folder if it does not exsist    
    # Make sure to pass the full folder name prior to executing this function
    if not os.path.exists(folder_name_with_path):
        os.makedirs(folder_name_with_path)
    else:
        pass
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def delete_and_make_folder(folder_name_with_path):
    """
    Parameters
    folder_name_with_path
    """
    # CORE FUNCTION
    # Recreates a folder everytime this function is invoked
    if not os.path.exists(folder_name_with_path):
        os.makedirs(folder_name_with_path)
    else:
        # First delete the exsisting folder and then recreate a fresh one
        shutil.rmtree(folder_name_with_path)
        os.makedirs(folder_name_with_path)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def ros_timestamp_to_int_timestamp(ros_msg):
    # CORE FUNCTION
    # Converts ros time into long integer timestamp string
    bx = None
    bx = int(ros_msg.header.stamp.to_time() * 1000) # sec.nanosec in float to long integer
    return bx
# ------------------------------------------------------------------------------------------




# ***************************** User Defined Functions *************************************


# ------------------------------------------------------ Main Code ----------------------------------------

"""
--> Define locations of "parent directory (i.e. SR1)"
--> By default, this script will work in the "robot0" directory. So create this directory if not exsist

"""

# Initialize parser
parser = argparse.ArgumentParser(description="Input bag file and dataset directory name")
parser.add_argument('rosbag_dir',type=str, help= 'Directory where rosbags are saved')
parser.add_argument('f_name',type=str, help= 'Name of bag file exlcuding .bag extension')
parser.add_argument('cam0',type=str, help= 'topic name of image messages')

# parser.add_argument('dataset_parent_dir',type=str, help= 'Parent folder that contains all the datasets')
# parser.add_argument('dataset_name',type=str, help= 'Directory which will contains the newely created dataset')

# parser.add_argument('robot0',type=str, help= 'Directory where robot_0 data is stored')

# parser.add_argument('imu0',type=str, help= 'chosen robot`s imu data')

args = parser.parse_args()

# Controls overall speed
loop_speed = 0.001

# Global list to hold timesteps separately, save this out as a txt file later on
cam0_tt_global_lss = []

# Frameskipper, define integer to store only the i^th frame in the sequence
frame_skip = 0 # Set -1 to save all frames
frame_cnt = 0 # Initialize
imgz_saved_cnt = 0 # Book keeper that records how many images were actually saved


# Initialize CvBridge object
br = CvBridge()

# Global flags
global ok_flag
ok_flag = False

## Echo passed arguments by user
print(f"------------------ Passed parameters -------------------")
print(f"Rosbag directory --> {args.rosbag_dir}")
print(f"Rosbag file name --> {args.f_name}")
print(f"Name of camera topic --> {args.cam0}")

print(f"------------------ Passed parameters -------------------")
print()

# Define directories
home_dir = str(Path.home()) # Location of /home folder
bag_main_dir = home_dir + "/" + args.rosbag_dir # Location of rosbag
script_dir = str(Path.cwd()) # https://stackoverflow.com/questions/5137497/find-the-current-directory-and-files-directory

# HARDCODED, name of folder to extract RGB images to
image_save_dir = script_dir + "/" + "data"

# Build file locations
input_bag_loc = bag_main_dir + "/" + args.f_name + ".bag"

# Create save directory
delete_and_make_folder(image_save_dir)

# Read the defined bag file
input_bag = rosbag.Bag(input_bag_loc)

# Separate out messages based on topic, each topic is filtered as a python generator object
cam0_bag = input_bag.read_messages(topics=args.cam0)

# Count number of cam0 messages
cam0_msg_count = input_bag.get_message_count(args.cam0)

# Flag to activate display of processed image
show_image_flag = True # Set false if you want to supress output

cv_win_wait_key = 1 # About 10ms of wait time

print("Processing......")

# Cycle through each images, extract timestep, and save images with timesteps
for ii in tqdm.trange(cam0_msg_count):
    # Define variables
    cam0_msg, cam0_t = [], 0
    imu0_msg, imu0_t = [], 0
    imu0_t_long_int = 0
    imu0_msg_line = [] # A list containing [timestep, vector.x, vector.y, vector.z]
    imgz_path = None # Location to save openCV image
    imgz_bgr = None # ros.Image converted to openCV numpy array
    
    # Read in data from the python generator (cam0_bag)
    _, cam0_msg, cam0_t = next(cam0_bag)
    
    # Push this timestep into the global list
    cam0_tt_global_lss.append(cam0_t)

    # Build the name of the image.png file
    imgz_path = image_save_dir + "/" + str(cam0_t) + ".png"

    # Convert image message to Numpy array for saving
    # Requires opencv-bridge ROS package
    imgz_bgr = convert_from_ros_to_opencv_imgz(cam0_msg, br)

    # Save in "data" directory only if it's the i^th frame of current count
    if (frame_cnt == frame_skip):
        #* Optional show imagepython
        if (show_image_flag):
            show_image(imgz_bgr, "Image", [640,480], [640,480], cv_win_wait_key)
        
        cv2.imwrite(imgz_path,imgz_bgr)
        imgz_saved_cnt = imgz_saved_cnt + 1
        frame_cnt = 0 # Reset
    else:
        frame_cnt = frame_cnt + 1

    # Delay to control speed
    time.sleep(loop_speed)

print()
print(f"----------------------- Debug Messages -----------------------------------------")
print(f"Done")
print(f"{frame_skip} frames/iteration skipped")
print(f"Out of {cam0_msg_count} image messages, {imgz_saved_cnt} were processed")
print(f"----------------------- Debug Messages -----------------------------------------")
print()

# Cleanup and release resources
cv2.destroyAllWindows() # Maybe redundant
input_bag.close()
#output_bag.close()
# ----------------------------------------------- EOF -----------------------------------------------------