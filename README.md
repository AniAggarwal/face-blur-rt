# Real Time Face Blur

A simple tool to provide real time blur of select faces on video input.

Check `design-document.md` for a timeline and details.

# Installation

1. Install conda or mamba
2. Create conda env for this project and install depencies: `conda env create --name face-blur-rt --file=face-blur-rt.yml`
3. Activate the env: `conda activate face-blur-rt`
4. Run the demo: `python main.py`


# Demo Video
https://github.com/AniAggarwal/face-blur-rt/assets/52252926/cabb24fc-01c7-495a-838c-ac7c18ac804a

# The Full Report
Final Project Report
Ani Aggarwal, Vance Degen, Varun Unnithan, Monish Napa, Rohit Kommuru

All code for the project can be found at: https://github.com/AniAggarwal/face-blur-rt

## Initial Goal
The basic project description is to use computer vision and object detection to blur faces in real time, i.e. face blurring on live footage. This tool would be useful for use cases such as live streaming in public, as well as potentially for the implementation of security footage and surveillance that can be monitored in real-time without compromising privacy. Some planned features are:
recognition system wherein specified faces remain unblurred
recurrent model to increase robustness of object detection against things like occlusion
give end user the choice between two different settings: higher frame rate and lower accuracy, or lower frame rate and higher accuracy
different blurring options, e.g. whole bounding box blur or facial segmentation blur

## Introduction

We are utilizing computer vision and object detection to perform effective real-time blurring of faces from both live video input and pre-recorded videos. This project has implications in streaming platforms such as Twitch to protect user privacy as well as in law enforcement where private video footage may need to be redacted through the protection of multiple suspects’ faces. While face blurring is an existing practice in these fields, it involves a “manual and time-consuming” process in which existing models “are subject to either excessive, inconsistent, or insufficient facial blurring” [6]. Given the degree of confidentiality required in the mentioned industries, it is important for facial blurring models, especially those created using machine learning, to yield optimal results. Furthermore, it is essential for face blurring to be fast and accurate, as a single frame can expose a person’s privacy to the public. This makes real time face blurring a difficult and important problem to solve.  Overall, our solution is to create a highly customizable architecture that utilizes efficient techniques and pre-existing face recognition libraries. For instance, our model regulates the amount of blurring by offering users flexibility through different blurring options such as the blurring shape and method as well as the frame rate and capturing method for face recognition. Additionally, the model offers robustness through a face tracker that retains a user’s face across a certain number of frames and avoids losing missing frames when there is movement and occlusion over the face, thus improving accuracy. 

## Literature review

### YuNet
YuNet [1] is an innovative face detection architecture that uses depthwise separable convolutions to decrease computational and parametric costs, and a specialized feature pyramid to efficiently detect both nearby and faraway faces. We used this mode for face detection due to its excellent balance between speed and accuracy, and its ability to be deployed on edge devices. 

### SCRFD
Sample and Computation Redistribution for Efficient Face Detection [2] is a method to produce high accuracy, low computation cost models. SCRFD involves two main techniques: Sampling Redistribution, which augments training samples for detection stages that need them most; and Computational Redistribution, which reallocates computational resources among the backbone, neck, and head of the model. Due to its promising results on benchmarks, we included this model as an alternative to YuNet.

### TinaFace
TinaFace [3] Implements a ResNet-50 based object detector, fine-tuned for the task of face detection. This model nets the highest benchmark scores on WIDER FACE, with 92.4% AP. We utilized this model to generate ground truth bounding boxes for our own benchmark video. This alleviated the time and energy needed to hand-draw bounding boxes, and allowed us to focus on more important tasks.

### SORT
Simple Online and Realtime Tracking [4] is a lightweight and efficient object tracking algorithm that uses Kalman filters and the Hungarian algorithm. We use the tracker to enhance the recall of our bounding boxes. Often, the detector can lose faces for several frames at a time, but SORT makes up for this by extrapolating where the faces may be and drawing bounding boxes accordingly. Note that since we value recall over precision, we have tuned the algorithm to be generous in its bounding box estimations, which sometimes results in boxes being drawn where no faces exist.

### SFace
Privacy-friendly and Accurate Face Recognition using Synthetic Data [5] is a technique to train face recognition models that, given an example face, generates synthetic faces as training data. This technique is both easy to set up (only requires a single picture upload), and privacy-preserving (due to the training data not using real faces). As our recognizer, we used a model trained on SFace. By allowing registered faces to remain unblurred, the facial recognition system enabled by SFace enables much functionality and allows us to deploy our program with few known face photos (in the order of 1s to 10s of photos).

## How libraries are used
Libraries such as NumPy and OpenCV were used to handle general image processing tasks, such as retrieving frames from the input and modifying each frame to include bounding boxes, blurs, and labels. PyTorch was used to pack and unpack pretrained models, and to get accurate timing (since CUDA is asynchronous) for benchmarking results. 

We chose YuNet and SCRFD as our FaceDetectors due to their high accuracy and speed in face detection. SFace Recognizer was chosen due to its ability to provide accurate face recognition with very little reference photos. Additionally, SORT was used as the tracking algorithm due to its lightweight architecture and efficiency. 


## Implementation/Project Architecture
A flow chart of the face blurring pipeline and architecture can be found in Figure A.

The class RealTimeFaceBlurrerByFrame is responsible for executing the entire pipeline on a specified video source. The class is initialized with the following parameters: video_source, face_recognition_model, face_detection_model, face_tracker, blur_method, blur_shape, and performance_settings. Each of these parameters correspond to its own class. In the main.py file, a RealTimeFaceBlurrerByFrame object is created using the specified parameters as well as options for video input, including either a path to a video file or live video stream. In the known-faces file, data is inputted containing images of faces that will be fed into the face recognizer and outputted accordingly. In the future, we might replace this with an auto configuration via a config file.

Classes FaceDetector, FaceRecognizer, and Blurrer are subclassed for specific algorithm implementations; For example, SCRFD and YuNet inherit FaceDetector (as seen in the chart under “Face Detection”). Each “Configurable” node in the graph indicates where users have a choice of which algorithm they plug into the pipeline [Fig. A]. We utilize abstract classes and inheritance to make options hot swappable and maintainable.

First, the input frame is fed into an instance of the FaceDetector class. The FaceDetector used returns the coordinates of bounding boxes and the features of each face detected. If the detector fails to detect faces for a few frames at a time, then the tracker can extrapolate where the bounding boxes should be and return these predicted bounding boxes.

The list of bounding boxes and their corresponding features are sent through an instance of the FaceRecognizer class. If the face is determined to be a known face, then the bounding box is labeled accordingly; otherwise, the chosen blur is added over the area of the bounding box. The final frame, with all labeled bounding boxes and blurs is then outputted to the user. 

This modular approach offers a high level of customizability; End users can choose each parameter according to their specific needs, and to achieve their desired balance between detection accuracy and framerate. Another advantage to this modular approach is that outside developers can easily add their own configurable options for each pipeline step. For example, while we have adapted two facial detection models- SCRFD and YuNet- that strike different balances between detection accuracy and framerate, one could easily train and insert more models to suit their purpose.

Overviews of each algorithm used are in the “Literature Reviews” section.

## Changes/Optimizations

An unexpected finding that arose was that the efficiency of the detectors turned out to be much better than the tracker. Our initial approach was to run the FaceDetector after a set amount of frames, and run the FaceTracker to draw the bounding boxes at every frame using the information provided by the FaceDetector. However, what we found was that deep pretrained face detection models ran much faster than trackers such as SORT. It is more efficient to run a FaceDetector at each frame than run the Face Detector and FaceTracker to retrieve the bounding boxes, as running the FaceDetector at each frame results in a higher frame rate compared to the FaceTracker. 

## Benchmarking/Testing

YuNet, an open–source library for face detection, has the following performance metrics on the WIDER dataset, a face detection benchmark dataset:
AP_easy=0.887, AP_medium=0.871, AP_hard=0.768, where AP = average prediction

SCRFD, a deep learning-based face detection model, has the following performance metrics, measured based on VGA resolution:
Easy: 96.06%, Medium: 94.92%, Hard: 85.29%, flops=34.13

Benchmarking for our model was performed by comparing the performance of TinaFace, an existing face detection method, with our own face detector on a 6 minute long clip recorded from https://youtu.be/TW_x7pzsK9I?t=17760 (from 04:56:00 to 05:02:00) in which metrics such as IoUs, boxes, and missed frames are captured from the footage of people’s faces. The video was downsampled from 60fps to 30fps with an input resolution of 1080p.

We chose a suitable real life Twitch stream for this to demonstrate our suggested use case. This stream clip includes portions with two main streamers walking by themselves, in larger crowds, and directly interacting with strangers. Additionally, this clip is chosen as there are numerous interactions in which stranger’s and the streamer’s faces are occluded. For example, a fan takes photos with this streamer in which she covers her face in the photo. This clip was also selected to include portions where there are a large number of faces directly visible on screen. For instance, from the 10 to 25 second marks upwards of 10 partially occluded faces or heads are on screen.

For reference, TinaFace performance metrics can be found in Figure C.

Both our program and the TinaFace baseline were run on a laptop with 12th Gen Intel(R) Core(TM) i7-12700H CPU and NVIDIA GeForce RTX 3050 Laptop GPU/PCIe/SSE2 GPU. TinaFace runs on GPU, while ours solely runs CPU. This computer was chosen as opposed to a compute cluster or large desktop to better mirror our suggested application. It is worth noting that our implementation runs at faster than real time on this system, taking an average of 6.8 milliseconds per frame, or an average FPS of 147.

The results of our model on detecting faces from the video are shown in Figure B.

IoU, or intersection over union, indicates the extent to which two facial recognition boxes overlap and is the area intersection of the two boxes divided by their union. Traditionally, a threshold IOU of 0.5 is used to mark bounding boxes as correct, but since we intentionally expand our bounding boxes after detection for privacy we choose a threshold of 0.4. In any given frame, there may be more than one bounding box detected by our baseline, but it is possible our model misses them. 

The first column represents the percentage of frames in the benchmark for which the IOU is greater than 0.4 for each bounding box we do not miss. The second column treats missed bounding boxes as an IOU of 0.0, greatly reducing the percentage. The third column shows that we miss a face in about half of the frames in the video, which is expected for such a difficult benchmark done in real time. The fourth and fifth columns are similar to the first two, but just compute the average IOUs. The last two columns are simple FPS and time per frame metrics. 
Our model performs faster than real time for all tasks except gaussian blur, which is a slow but aesthetic blur (though looks nearly identical to box blur).

Figure D captures the average IoUs recorded by our model across the frames of the video as compared to TinaFace in terms of bounding boxes. The drop between 10 and 25 seconds is due to the large number of faces in frame in the benchmark at that time.

Figure E depicts the frame rate of our model and the TinaFace baseline. Our model performs far faster when not running recognition, the main bottleneck. However, black and box blurs and little overhead. Worth noting that TinaFace crashed while benchmarking after being loaded, detecting the large number of faces in that 10 to 25 second period. We chose to leave this in to demonstrate the unreliability of such models and infeasibility for real time purposes. The spikes when recognition is enabled is when there are no faces in frame and thus no recognition and blurring to do.

Figure F depicts the number of excess bounding boxes drawn per frame, with a negative number indicating missed faces in that frame. Note that with the tracking algorithm enabled, our system tends to draw more excess bounding boxes; this is because we have the tracker tuned to extrapolate the position for several frames after the detector loses a face. This behavior lets the tracker keep faces covered when the detector produces a false negative, but also sometimes results in bounding boxes being drawn where no faces exist. As you can see our tuning was effective, with the tracker consistently outperforming the detector in this metric.

## Discussion

Overall, our initial goals were met. We were successfully able to create an effective face blurring model that performs optimal face detection and blurring on both live input and recorded videos with multiple faces. The recognition model produces results with relatively high accuracy and successfully not only performs blurring with different user customization options but also outputs the closest matching name corresponding to the face. However, we were unable to create an advanced user interface where users could manually select which faces to unblur due to the time frame of the project and the complexity of integrating this system with the existing recognition model. 

There are a few limitations that could be resolved involving the scope of the project. In future implications, the project could be expanded to use more advanced algorithms to accurately detect and blur faces in situations such as rapid movement, low frame rate, or occluded faces where the model may not perform best. Additionally, the tracking algorithm could be tweaked to decrease erroneous bounding boxes, while still maintaining a high recall of faces covered, and the face recognizer could be optimized to distinguish between faces that appear similar. Finally, the model could be more computationally efficient in order to process larger streams of data, such as hours of pre-recorded footage, which may require significant time to run.


## Contributions by Members
### Ani

I worked out an outline for the project, a timeline, project expectations and deliverables. I later wrote the code for the MVP and spent time ensuring that it was written in an OOP manner to facilitate better collaboration (e.g. avoiding merge conflicts). I then read papers and experimented with different models and tried training my own (compute limited on my laptop and was refused extra compute) and settled on two different pretrained models for face detection. Did a similar process with recognition but only choose one. Overall, to deliver the MVP and a bit more I wrote the main controller, face detector, recognizer, blurrer, and various options for a total of about 2,000 lines. I then wrote a benchmarking suite which included researching and getting TinaFace running, modifying vedadet’s code to run TinaFace on a video and output metrics to a file. I then added benchmarking features to our project and created a Jupyter notebook to take the results from the CSVs and create graphs and tables. This process took a lot of back and forth and iterations to make everything work and look nice, and running benchmarks took a long time. I also wrote about all of the benchmarking in the final report. In addition to the code I wrote I worked as a project manager and scrum master in an effort to encourage progress to be made and did my best to support other team members to reduce the friction needed to work on the project.

### Vance

I tested alternatives to YuNet and SCRFD, namely BlazeFace, ULFG,  and Haar Cascade. My main contribution was in face tracking. I tested several tracking algorithms, such as correlational tracking, CSRT, KCF, MOSSE, and wrote my own bounding box interpolation algorithm based on cubic spline interpolation. I ultimately decided that the SORT algorithm exceeded all other techniques in both speed and accuracy. Due to the SORT algorithm not offering enough of an improvement over the existing detection framework, I tweaked the implementation to offer a high recall, which leads to a notably higher coverage of faces. 

### Varun

I worked to research and learn more about the systems and architectures with the project, and experimented with using a custom Kalman Filter for face tracking, though was ultimately unable to achieve a usable version of it to work. I also tested and wrote different blurring algorithms to optimize the real-time blurrer for speed and effectiveness. I helped write the report and system architecture figures.

### Rohit

I worked on the debugging issues with face tracker, recognizer, and detector. To do this, I experimented with the model using different reference photos and different testing. I found issues with the retrieval of facial features from the face detection model and tracker, and the input and output resolution of the face detection and recognizer. Through experimentation, I set a cosine similarity threshold to determine whether or not a face should be recognized. Additionally, I worked on drawing the bounding boxes and labeling.  

### Monish

I worked on improving coverage of the face blurrer by expanding the detection area of 
each box covered by the face detector algorithm and updating the circular blurring shape 
to account for more elliptical features using calculations involving the dimensions of the 
box. This allows the gaussian blurring algorithm as well as the circular bounding box to 
reflect the shapes of users’ faces more closely. In addition, I researched existing blurring 
methods, performed testing of the facial recognition algorithm, and adjusted the elliptical 
detection to account for different angles. Finally, I helped construct major sections of 
the report and created the presentation/poster material.

**Output of modified git-log command for reference:**
```
AniAggarwal,: 152 files changed, 199934 insertions(+), 33822 deletions(-), 166112 net
monishnapa,: 1 files changed, 10 insertions(+), 8 deletions(-), 2 net
noreply,: 9 files changed, 170 insertions(+), 6 deletions(-), 164 net
rohitkommuru,: 7 files changed, 110 insertions(+), 71 deletions(-), 39 net
vancedegen,: 76 files changed, 36784 insertions(+), 351 deletions(-), 36433 net
vancedegen,noreply,: 1 files changed, 4 insertions(+), 8 deletions(-), -4 net
varun.unnithan33,: 2 files changed, 19 insertions(+), 4 deletions(-), 15 net
varun.unnithan33,noreply,: 2 files changed, 11 insertions(+), 2 deletions(-), 9 net
```

## Figures

A. Face blurring pipeline
![pipeline](https://github.com/AniAggarwal/face-blur-rt/assets/52252926/fb32eae0-7916-49bd-bea3-c8dc2325bd29)


B. Our solution benchmark compared to TinaFace.
![statistics_table](https://github.com/AniAggarwal/face-blur-rt/assets/52252926/d18f5e73-af15-40e9-ac4c-dce1b0577c9c)


C. TinaFace benchmarks.

| Model | Size | AP50(VOC12) | WIDERFACE Easy | WIDERFACE Medium | WIDERFACE Hard |
| ----- | ---- | ---------- | -------------- | --------------- | ------------- |
| TinaFace R50-FPN-GN-DCN | (1100, 1650) | 0.923 | 0.963 | 0.957 | 0.930 |


D.  Average intersection over Union, using TinaFace bounding boxes as ground truth.
![average-iou-by-time](https://github.com/AniAggarwal/face-blur-rt/assets/52252926/ed0c5e1d-350d-438b-bd21-4c7150d35181)


E. Performance analysis of pipeline; Time vs FPS, and Time vs Milliseconds per Frame (compared to TinaFace).
![fps-ms-combined-all](https://github.com/AniAggarwal/face-blur-rt/assets/52252926/0eda9879-2873-4a45-9e77-b5f4a03bf541)

F. Number of excess bounding boxes, using TinaFace bounding boxes as ground truth.
![excess-bboxes-by-time-bin](https://github.com/AniAggarwal/face-blur-rt/assets/52252926/9f83abd8-1428-4f28-92c5-912aa724eb1e)

## Bibliography:

Wu, W., Peng, H. & Yu, S. YuNet: A Tiny Millisecond-level Face Detector. Mach. Intell. Res. 20, 656–665 (2023).
Guo, Jia, et al. "Sample and computation redistribution for efficient face detection." arXiv preprint arXiv:2105.04714 (2021).
Zhu, Yanjia, et al. "Tinaface: Strong but simple baseline for face detection." arXiv preprint arXiv:2011.13183 (2020).
Bewley, Alex, et al. "Simple online and realtime tracking." 2016 IEEE international conference on image processing (ICIP). IEEE, 2016.
Boutros, Fadi, et al. "Sface: Privacy-friendly and accurate face recognition using synthetic data." 2022 IEEE International Joint Conference on Biometrics (IJCB). IEEE, 2022.
Bajpai, R., Aravamuthan, B. “SecurePose: Automated face blurring and human movement kinematics extraction from videos recorded in clinical settings.” arXiv
preprint arXiv:2402.14143 (2024)
