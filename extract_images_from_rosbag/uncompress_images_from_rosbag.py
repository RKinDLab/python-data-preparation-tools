# Script to extract compressed RGB images from ROSBAGS
# Author: Azmyin Md. Kamal
# Date : 04/04/2023
# Version: 1.5

"""
This version converts an uncompressed image to an image file

Requirements
** ROSBAGS directory must be in the home directory folder
** > Python 3.6 needed

Useage
--Directory of rosbags, name of bag file, name of dataset directory, camera topic name
-- "dataset_dir" argument should be passed in the following format "LSU_iCORE/experiment_name i.e LSU_iCORE/SR1
-- Put the camera_calib.yaml file manually once the dataset is created

Changelog
* 04/01/23 - Initial version
* 04/03 - Frameskipper functionality added to speed up dataset

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
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import rosbag # ROS library to manipulate rosbags 
from cv_bridge import CvBridge, CvBridgeError # Library to convert image messages to numpy array 


# --------------------------------------------------------------------------------------------------------
"""
# Source https://gist.github.com/awesomebytes/958a5ef9e63821a28dc05775840c34d9
ROS ImageTools, a class that contains methods to transform
from & to ROS Image, ROS CompressedImage & numpy.ndarray (cv2 image).
Also deals with Depth images, with a tricky catch, as they are compressed in
PNG, and we are here hardcoding to compression level 3 and the default
quantizations of the plugin. (What we use in our robots).
Meanwhile image_transport has no Python interface, this is the best we can do.
Author: Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
"""

# ***************************** ImageTools class definition *************************************

class ImageTools(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

    def convert_ros_msg_to_cv2(self, ros_data, image_encoding='bgr8'):
        """
        Convert from a ROS Image message to a cv2 image.
        """
        try:
            return self._cv_bridge.imgmsg_to_cv2(ros_data, image_encoding)
        except CvBridgeError as e:
            if "[16UC1] is not a color format" in str(e):
                raise CvBridgeError(
                    "You may be trying to use a Image method " +
                    "(Subscriber, Publisher, conversion) on a depth image" +
                    " message. Original exception: " + str(e))
            raise e

    def convert_ros_compressed_to_cv2(self, compressed_msg):
        np_arr = np.fromstring(compressed_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def convert_ros_compressed_msg_to_ros_msg(self, compressed_msg,
                                              encoding='bgr8'):
        cv2_img = self.convert_ros_compressed_to_cv2(compressed_msg)
        ros_img = self._cv_bridge.cv2_to_imgmsg(cv2_img, encoding=encoding)
        ros_img.header = compressed_msg.header
        return ros_img

    def convert_cv2_to_ros_msg(self, cv2_data, image_encoding='bgr8'):
        """
        Convert from a cv2 image to a ROS Image message.
        """
        return self._cv_bridge.cv2_to_imgmsg(cv2_data, image_encoding)

    def convert_cv2_to_ros_compressed_msg(self, cv2_data,
                                          compressed_format='jpg'):
        """
        Convert from cv2 image to ROS CompressedImage.
        """
        return self._cv_bridge.cv2_to_compressed_imgmsg(cv2_data,
                                                        dst_format=compressed_format)

    def convert_ros_msg_to_ros_compressed_msg(self, image,
                                              image_encoding='bgr8',
                                              compressed_format="jpg"):
        """
        Convert from ROS Image message to ROS CompressedImage.
        """
        cv2_img = self.convert_ros_msg_to_cv2(image, image_encoding)
        cimg_msg = self._cv_bridge.cv2_to_compressed_imgmsg(cv2_img,
                                                            dst_format=compressed_format)
        cimg_msg.header = image.header
        return cimg_msg

    def convert_to_cv2(self, image):
        """
        Convert any kind of image to cv2.
        """
        cv2_img = None
        if type(image) == np.ndarray:
            cv2_img = image
        elif image._type == 'sensor_msgs/Image':
            cv2_img = self.convert_ros_msg_to_cv2(image)
        elif image._type == 'sensor_msgs/CompressedImage':
            cv2_img = self.convert_ros_compressed_to_cv2(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return cv2_img

    def convert_to_ros_msg(self, image):
        """
        Convert any kind of image to ROS Image.
        """
        ros_msg = None
        if type(image) == np.ndarray:
            ros_msg = self.convert_cv2_to_ros_msg(image)
        elif image._type == 'sensor_msgs/Image':
            ros_msg = image
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_msg = self.convert_ros_compressed_msg_to_ros_msg(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_msg

    def convert_to_ros_compressed_msg(self, image, compressed_format='jpg'):
        """
        Convert any kind of image to ROS Compressed Image.
        """
        ros_cmp = None
        if type(image) == np.ndarray:
            ros_cmp = self.convert_cv2_to_ros_compressed_msg(
                image, compressed_format=compressed_format)
        elif image._type == 'sensor_msgs/Image':
            ros_cmp = self.convert_ros_msg_to_ros_compressed_msg(
                image, compressed_format=compressed_format)
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_cmp = image
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_cmp

    def convert_depth_to_ros_msg(self, image):
        ros_msg = None
        if type(image) == np.ndarray:
            ros_msg = self.convert_cv2_to_ros_msg(image,
                                                  image_encoding='mono16')
        elif image._type == 'sensor_msgs/Image':
            image.encoding = '16UC1'
            ros_msg = image
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_msg = self.convert_compressedDepth_to_image_msg(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_msg

    def convert_depth_to_ros_compressed_msg(self, image):
        ros_cmp = None
        if type(image) == np.ndarray:
            ros_cmp = self.convert_cv2_to_ros_compressed_msg(image,
                                                             compressed_format='png')
            ros_cmp.format = '16UC1; compressedDepth'
            # This is a header ROS depth CompressedImage have, necessary
            # for viewer tools to see the image
            # extracted from a real image from a robot
            # The code that does it in C++ is this:
            # https://github.com/ros-perception/image_transport_plugins/blob/indigo-devel/compressed_depth_image_transport/src/codec.cpp
            ros_cmp.data = "\x00\x00\x00\x00\x88\x9c\x5c\xaa\x00\x40\x4b\xb7" + ros_cmp.data
        elif image._type == 'sensor_msgs/Image':
            image.encoding = "mono16"
            ros_cmp = self.convert_ros_msg_to_ros_compressed_msg(
                image,
                image_encoding='mono16',
                compressed_format='png')
            ros_cmp.format = '16UC1; compressedDepth'
            ros_cmp.data = "\x00\x00\x00\x00\x88\x9c\x5c\xaa\x00\x40\x4b\xb7" + ros_cmp.data
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_cmp = image
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_cmp

    def convert_depth_to_cv2(self, image):
        cv2_img = None
        if type(image) == np.ndarray:
            cv2_img = image
        elif image._type == 'sensor_msgs/Image':
            image.encoding = 'mono16'
            cv2_img = self.convert_ros_msg_to_cv2(image,
                                                  image_encoding='mono16')
        elif image._type == 'sensor_msgs/CompressedImage':
            cv2_img = self.convert_compressedDepth_to_cv2(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return cv2_img

    def convert_compressedDepth_to_image_msg(self, compressed_image):
        """
        Convert a compressedDepth topic image into a ROS Image message.
        compressed_image must be from a topic /bla/compressedDepth
        as it's encoded in PNG
        Code from: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/
        """
        depth_img_raw = self.convert_compressedDepth_to_cv2(compressed_image)
        img_msg = self._cv_bridge.cv2_to_imgmsg(depth_img_raw, "mono16")
        img_msg.header = compressed_image.header
        img_msg.encoding = "16UC1"
        return img_msg

    def convert_compressedDepth_to_cv2(self, compressed_depth):
        """
        Convert a compressedDepth topic image into a cv2 image.
        compressed_depth must be from a topic /bla/compressedDepth
        as it's encoded in PNG
        Code from: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/
        """
        depth_fmt, compr_type = compressed_depth.format.split(';')
        # remove white space
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()
        if compr_type != "compressedDepth":
            raise Exception("Compression type is not 'compressedDepth'."
                            "You probably subscribed to the wrong topic.")

        # remove header from raw data, if necessary
        if 'PNG' in compressed_depth.data[:12]:
            # If we compressed it with opencv, there is nothing to strip
            depth_header_size = 0
        else:
            # If it comes from a robot/sensor, it has 12 useless bytes apparently
            depth_header_size = 12
        raw_data = compressed_depth.data[depth_header_size:]

        depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8),
                                     # the cv2.CV_LOAD_IMAGE_UNCHANGED has been removed
                                     -1)  # cv2.CV_LOAD_IMAGE_UNCHANGED)
        if depth_img_raw is None:
            # probably wrong header size
            raise Exception("Could not decode compressed depth image."
                            "You may need to change 'depth_header_size'!")
        return depth_img_raw

    def display_image(self, image):
        """
        Use cv2 to show an image.
        """
        cv2_img = self.convert_to_cv2(image)
        window_name = 'show_image press q to exit'
        cv2.imshow(window_name, cv2_img)
        # TODO: Figure out how to check if the window
        # was closed... when a user does it, the program is stuck
        key = cv2.waitKey(0)
        if chr(key) == 'q':
            cv2.destroyWindow(window_name)

    def save_image(self, image, filename):
        """
        Given an image in numpy array or ROS format
        save it using cv2 to the filename. The extension
        declares the type of image (e.g. .jpg .png).
        """
        cv2_img = self.convert_to_cv2(image)
        cv2.imwrite(filename, cv2_img)

    def save_depth_image(self, image, filename):
        """
        Save a normalized (easier to visualize) version
        of a depth image into a file.
        """
        # Duc's smart undocummented code
        im_array = self.convert_depth_to_cv2(image)
        min_distance, max_distance = np.min(im_array), np.max(im_array)
        im_array = im_array * 1.0
        im_array = (im_array < max_distance) * im_array
        im_array = (im_array - min_distance) / max_distance * 255.0
        im_array = (im_array >= 0) * im_array

        cv2.imwrite(filename, im_array)

    def load_from_file(self, file_path, cv2_imread_mode=None):
        """
        Load image from a file.
        :param file_path str: Path to the image file.
        :param cv2_imread_mode int: cv2.IMREAD_ mode, modes are:
            cv2.IMREAD_ANYCOLOR 4             cv2.IMREAD_REDUCED_COLOR_4 33
            cv2.IMREAD_ANYDEPTH 2             cv2.IMREAD_REDUCED_COLOR_8 65
            cv2.IMREAD_COLOR 1                cv2.IMREAD_REDUCED_GRAYSCALE_2 16
            cv2.IMREAD_GRAYSCALE 0            cv2.IMREAD_REDUCED_GRAYSCALE_4 32
            cv2.IMREAD_IGNORE_ORIENTATION 128 cv2.IMREAD_REDUCED_GRAYSCALE_8 64
            cv2.IMREAD_LOAD_GDAL 8            cv2.IMREAD_UNCHANGED -1
            cv2.IMREAD_REDUCED_COLOR_2 17
        """
        if cv2_imread_mode is not None:
            img = cv2.imread(file_path, cv2_imread_mode)
        img = cv2.imread(file_path)
        if img is None:
            raise RuntimeError("No image found to load at " + str(file_path))
        return img
# --------------------------------------------------------------------------------------------------------


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

# Frameskipper, define integer to store only the i^th frame in the sequence
frame_skip = 2 # Need to test it
frame_cnt = 0 # Initialize
imgz_saved_cnt = 0 # Book keeper that records how many images were actually saved

# Global list to hold timesteps separately, save this out as a txt file later on
cam0_tt_global_lss = []

# Initialize CvBridge object
br = CvBridge()

# Initialize Image tools
it = ImageTools()

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
    # Requires opencv-bridge ROS package (works for uncompressed image)
    # imgz_bgr = convert_from_ros_to_opencv_imgz(cam0_msg, br)
    imgz_bgr = it.convert_ros_compressed_to_cv2(cam0_msg)

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
print(f"Out of {cam0_msg_count} image messages, {imgz_saved_cnt} were processed")
print(f"----------------------- Debug Messages -----------------------------------------")
print()

# Cleanup and release resources
cv2.destroyAllWindows() # Maybe redundant
input_bag.close()
#output_bag.close()
# ----------------------------------------------- EOF -----------------------------------------------------