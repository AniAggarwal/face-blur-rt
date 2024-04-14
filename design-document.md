# Python System Design Document: Real-Time Face Blurring Tool

## TODO:

- [x] FPS counter - Ani
- [x] Face recognition (that works) - Rohit
- [ ] Add ellipse mode to better cover face with circular mode - Monish
- [ ] Improve performance of blurring techniques (except for black square, which is very fast)
    - parituclarly guassian blur with circle
- [ ] Add haars cascade face detection - Vance
- [ ] Create a temporal tracking model - Varun
- [ ] Create custom frame by frame DNN
- [x] Add an sample video and code to read from video instead of just camera
 
## Overview

The project aims to develop a Python-based tool for real-time face blurring in live footage, ideal for livestreaming in public spaces. This tool will incorporate computer vision and object detection techniques to identify and blur faces dynamically. Key features will include selective face recognition to leave specified faces unblurred, a recurrent model to handle occlusions effectively, and customizable settings for balancing frame rate and accuracy. Additional blurring options will also be provided.

### Objectives

- **Privacy in Public Livestreams:** Ensure individuals' privacy by blurring faces in real-time during livestreams in public areas.
- **Selective Recognition:** Implement a system to keep certain faces unblurred based on a recognition system.
- **Customization:** Offer users the ability to choose between performance or accuracy and select their preferred blurring technique.
- **Scalability and Flexibility:** Design the system to easily incorporate future enhancements and maintain high performance.

## System Architecture

### Core Components

1. **Video Input Handler:** Manages live video feeds from webcams or pre-recorded footage, ensuring compatibility across different devices and formats.
2. **Face Detection Module:** Utilizes OpenCV to detect faces in each video frame. This module will start with an out-of-the-box model and evolve to a custom, speed-optimized model.
3. **Blurring Techniques:** Implements various blurring methods, including bounding box and facial segmentation blurring, to obscure detected faces.
4. **Face Recognition Module:** Identifies specific faces to remain unblurred, using a pre-trained model with the capability to learn and recognize new faces over time.
5. **Performance Settings:** Allows users to toggle between high frame rate with lower accuracy or lower frame rate with higher accuracy.
6. **Temporal Face Tracking:** Enhances detection accuracy by tracking faces across frames, reducing the need for constant detection and improving performance.

### Optional Features

- **Recurrent Model for Robust Detection:** Implements a recurrent neural network (RNN) or similar technology to improve detection robustness, especially in cases of partial occlusion.
- **Advanced User Interface:** Develops an intuitive UI for live settings adjustments, including selecting which faces to exempt from blurring and changing performance modes.

## Technology Stack

- **Programming Language:** Python 3.8+
- **Computer Vision:** OpenCV 4.x for initial face detection and video processing.
- **Machine Learning Framework:** TensorFlow or PyTorch for developing the recurrent model and custom face detection models.
- **Development Tools:** Git for version control, GitHub for repository hosting, and Docker for containerization to ensure environment consistency.

## Development Timeline

1. **Week 1-2: Setup**

   - Initialize Git repository.
   - Set up development environment with Python and OpenCV.
   - Implement webcam video capture using OpenCV.

2. **Week 3-4: Basic Face Detection**

   - Integrate an out-of-the-box model for simple face detection in still images.
   - Apply basic blurring over detected faces.

3. **Week 5-6: Video Processing and Facial Recognition**

   - Extend face detection to video streams, processing frames in real-time.
   - Incorporate a basic facial recognition system to identify specific individuals.

4. **Week 7-8: MVP and Performance Enhancement**

   - Combine face detection and recognition into a seamless workflow.
   - Implement temporal face tracking for improved accuracy.
   - Develop the system to work efficiently with video by detecting faces every n frames and tracking them.

5. **Week 9-10: Custom Model Development**

   - Start developing a custom face detection model focused on speed to replace the out-of-the-box solution.
   - Test and integrate the recurrent model for handling occlusions.

6. **Week 11-12: Finalization and Documentation**
   - Refine the system, focusing on performance optimization and user customization options.
   - Complete all documentation, including installation instructions and user guides.
   - Prepare the GitHub repository for final submission.

## Scalability and Growth

The system design prioritizes modularity and scalability. The separation of core components allows for independent updates and improvements. For instance, the face detection module can be upgraded without altering the video input handler or blurring techniques. The use of containerization and a well-defined API between components ensures that the system can scale with additional features, such as new blurring methods or enhanced recognition capabilities.

To accommodate growth, the project will follow best practices in code documentation, version control, and testing. This approach facilitates future contributions and the integration of new technologies, ensuring the tool remains at the forefront of privacy tech in live broadcasting.
