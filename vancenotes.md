- delay source video by a few hundred frames (static value), let this delayed footage be the buffer
- calculate bboxes every n frames of the buffer
- n should probably be static
- interpolate bbox location for uncalcuated frames 
- This method increases FPS by decreasing the number of calls to the detection->recognition->blur pipeline
- problem: cannot process batches and display to screen at same time

- WITH MULTITHREADING:
- one thread processes batches and adds to buffer, other thread displays completed frames from buffer

- concept: add a swapchain and only render frames that are complete
- https://www.youtube.com/watch?v=YNFaOnhaaso&list=PLii79NqsUd-6uBY4WC2Qva41l3cmYxFop&index=2
- place block = perform pipeline

- NOTES ON INTERPOLATION:
- process batch_size frames per batch, and only run pipeline on every frame_interval frames
- assert batch_size % frame_interval = 0

-   [[x1, y1, x2, y2]   face 1
     [x1, y1, x2, y2]]  face 2


- face tracking