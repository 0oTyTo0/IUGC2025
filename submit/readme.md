# Submission Guidelines

This folder contains the template for your submission. Please follow these guidelines to ensure your submission is valid:

## Submission Structure
Your submission should maintain the following structure:
- `metadata`：just copy this
- `model.pth`: Your trained model weights (Note: In this template, model.pth is empty due to file size limitations. Please replace it with your actual model weights when submitting)
- `model.py`: Your model implementation code
**model.py file ： load and predict methods of the model class need to be defined in it, participants should pay attention to the input and output requirements while designing. We do not recommend the use of third-party libraries**
## Submission Requirements
1. Make sure all your code and model files are included according to the structure above
2. Ensure your model.py contains all necessary functions for model loading and inference
3. Replace the empty model.pth with your actual trained model weights
submit a zip in the coadbench platform.

xxxx.zip/  
  ── metadata 
  ── model.py
  ── xxx.pth

## IMPORTANT: Input and Output Specification
 ----------------------------------------
 **Input (X)**:   
   - A PIL.Image object (from PIL.Image.open)
  - Represents an RGB image 
 
 **Output (coords)**:
   - A numpy array of shape (6,) containing 3 keypoint coordinates
   - Format: [x1, y1, x2, y2, x3, y3]
   - Coordinates must be in the pixel space of the original input image（512*512 here）
   - The coordinates should represent the exact locations of the detected keypoints
Thank you for following the submission guidelines!

## you need to define a class named "model", which defines "predict" and "load" functions. The codabench will automatically call these functions to get results.

