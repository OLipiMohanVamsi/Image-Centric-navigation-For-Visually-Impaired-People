
# Image-Centric Indoor Navigation Solution for Visually Impaired People

Navigation in indoor environments is highly challenging for a visually impaired person, 
particularly in an unknown environment.
For this purpose, we propose an image-centric indoor navigation solution using state-of-the-art computer vision techniques. 

In this project, we employ YOLOv8 as the object detection model for real-time processing of visual data to assist visually impaired persons during navigation. It analyzes images coming from a forward-facing camera and identifies objects and obstacles within the environment and produces intuitive navigational cues, for example, directional guidance to move "left" or "right" relative to the central point of the image. 

We use the COCO dataset (Common Objects in Context) for the training and fine-tuning of 
YOLOv8, which will enable it to robustly detect a very wide range of objects found in indoor 
environments. This ensures that the system is able to identify furniture, doors, pathways, and 
other features that are important for safe navigation. The objects that are detected are 
categorized and positioned on a spatial map, allowing the system to infer optimal paths for 
movement. Navigation instructions are dynamically generated based on the relative position of 
objects and communicated to the user through audio feedback or haptic devices. 


Some of the key innovations of this solution are its lightweight design for real-time processing, 
high object detection accuracy, and adaptability to different indoor settings. With YOLOv8, it 
achieves a good balance between computational efficiency and detection performance, so the 
system is suitable for deployment on portable devices like smartphones or wearable gadgets. 
In addition, integration with spatial analysis algorithms will provide users with precise and 
context-aware guidance. 

Thorough testing was performed on the simulated and real-world indoor environment to check 
the usability of the system. The results show that the presented solution will substantially 
enhance mobility and independence in the life of visually impaired users, offering timely and 
reliable navigation support. The next stages of the work include expansion of the model through 
incorporating depth perception, semantic segmentation, and support for personalized indoor 
maps, which are essential to make the system even more versatile and accurate. 

In conclusion, this project is a step forward in assistive technology that utilizes the power of 
deep learning and computer vision to empower visually impaired individuals in navigating 
complex indoor environments with confidence and safety. 

Keywords: Indoor navigation, visually impaired, YOLOv8, object detection, COCO dataset, 
real-time guidance, assistive technology, spatial mapping, mobility aid, computer vision.

