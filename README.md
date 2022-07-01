# deepdreamvideo
This project takes in a local video and "dreamifies" each frame individually. It also compiles the images into a video, but my preferred technique is to import the frames into a video editing software. No audio.

Variables to play with:
frameRate - self explanatory
max_dim - Width of output video in pixels
names - Layers of DeepDream model to activate. 'mixed0' through 'mixed10' create slightly different effects. 
octaveScale - size of dreamified objects. Do you want the dog faces small or large?
stepSize - the larger the step, the more trippy it is

Developed from Google's open source code for Deep Dream technique. 
