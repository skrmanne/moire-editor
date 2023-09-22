"""
* Editing a moire image in frequency domain.
* Author: Sai Kumar Reddy Manne
* Date : 09/21/2023
* Class: CS 7180 Advanced Perception
"""

#imports
import os, sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

def low_pass_filter(img, r):
    """ Low pass filter which zeros out anything outside a square of size r.
    """
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-r:crow+r, ccol-r:ccol+r] = 1
    return mask

def high_pass_filter(img, r):
    """High pass filter which zeros out anything inside a square of size r.
    """
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
    return mask

def getFFT(img, show=False):
    """Returns the FFT of the input image.
    """
    fft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft = np.fft.fftshift(fft)

    # display
    if show:
        mag = 20*np.log(cv2.magnitude(fft[:,:,0], fft[:,:,1]))
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("magnitude spectrum:", mag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return fft


def checkFFT():
    """Moire patterns show similar frequency distribution across all three channels.
    Hence, identifying and filtering out the moire patterns in one channel should be sufficient.
    """

    img = cv2.imread("input5.jpeg")
    # get three channel FFT
    fftR, fftG, fftB = getFFT(img[:,:,0], True), getFFT(img[:,:,1], True), getFFT(img[:,:,2], True)

def restore_moire_v1(rgb, r=250):
    """Simple moire restoration using a low pass filter. Not very effective for low frequency moire patterns.
    Removes more high frequency details than just the moire patterns.
    For 3 channel images, we do the FFT->filter->inverse FFT for each channel separately.
    Expects a fixed radius for the filter size given by the user.
    """
    restored_img = np.zeros(rgb.shape, np.uint8)
    for i in range(3):
        img = rgb[:,:,i]
        fft = getFFT(img)

        # filter and inverse FFT
        mag = 20*np.log(cv2.magnitude(fft[:,:,0], fft[:,:,1]))
        fshift = fft*low_pass_filter(mag, r=r)  # filtering FFT

        # inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)

        # bring the real, complex FFT to single channel magnitude and normalize
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        restored_img[:,:,i] = img_back

    return restored_img

def filterFFT(mask):
    """Uses a pre-defined mask to filter the FFT. Expands the mask to match input image shape.
    Mask can come from iterative edits by a user.
    """
    # create a broadcasted version of mask to match fft
    mask = np.tile(np.expand_dims(mask, 2), (1,1,2))
    restored_img = np.zeros(original.shape, np.uint8)

    for i in range(3):
        img = original[:,:,i]
        fft = getFFT(img)
        fshift = fft*mask

        # inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)

        # move 2 channel FFT to real normalized single channel space.
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        restored_img[:,:,i] = img_back
    return restored_img


# global variable used for mouse event tracking and drawing on the interactive editor.
x0, y0 =  -1, -1
draw = False

def draw_filter(event, x, y, flags, param):
    """Draws a rectangle on the power spectrum to create a band-stop filter in the drawn region.
    Uses left mouse events to track the area to draw a rectangle.
    """
    global x0, y0, draw, rgb

    if event == cv2.EVENT_LBUTTONDOWN:
        x0, y0, draw = x, y, True
    elif event == cv2.EVENT_MOUSEMOVE and draw:
        cv2.rectangle(mag, (x0, y0), (x, y), (0, 0, 0), 3)
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        cv2.rectangle(mag, (x0, y0), (x, y), (0, 0, 0), -1)

        # filter the image with updated mask
        rgb = filterFFT(mag)

def restore_moire_v2():
    """Create a simple mouse click based band-stop filter for the FFT.
    When a rectangle is drawn, fft is filtered and image is updated.
    Note: User can draw rectangles until desired by dragging the mouse with left button pressed.
    Application can be terminated by pressing 'c' key.
    """
    cv2.namedWindow("Frequency editor")
    cv2.setMouseCallback("Frequency editor", draw_filter)

    while True:
        mag_rgb = np.concatenate(
            (np.tile(np.expand_dims(mag, 2), (1,1,3)), rgb, original), axis=1)

        cv2.imshow("Frequency editor", mag_rgb)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break
    cv2.destroyAllWindows()

# main
parser = argparse.ArgumentParser(
    description="Moire pattern removal using FFT and band-stop filters. Usage: python3 moire.py --input <path-to-image>")
parser.add_argument("--input", type=str, help="Path to input image.")
args = parser.parse_args()

img = cv2.imread(args.input)
original = img.copy()   # keep a copy of original image to display side by side with restored image.
rgb = img.copy()    # copy of the original image updated with each filter iteration.

# convert to grayscale and get FFT for mask creation and editing.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
fft = np.fft.fftshift(fft)
mag = 20*np.log(cv2.magnitude(fft[:,:,0], fft[:,:,1]))
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

restore_moire_v2()
cv2.imwrite(args.input.replace(".", "-out."), rgb)