### PanoramaApp
An easy Panorama App that will create a panaroma picture from 3 adjacent pictures using Homography, RANSAC, Bacward Wrap, and Blend.

## Installation Instructions

You must have Python and OpenCV (`cv2`) installed on your computer. Follow the steps below:

### 1. Install Python
If you don’t have Python installed, you can download it from the [official Python website](https://www.python.org/downloads/). After downloading, follow the installation instructions for your platform.

To verify the installation, run the following command in your terminal:
```python
python3 --version
```
or
```python
python --version
```

### 2. Install OpenCV ('cv2')
```bash
pip3 install opencv-python
```
To verify OpenCV was installed correctly, you can check by running the following command in Python:

```python
import cv2
print(cv2.__version__)
```
### Usage
Provide a folder with your pictures naming them "1.png, 2.png, 3.png" in the same order you'd like them to be stitched and place the folder in the same directory as the code files.

