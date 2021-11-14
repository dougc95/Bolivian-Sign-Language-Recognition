# Hand Gesture Recognition for Bolivian Sign Language
This project aims to be the first step towards Dynamic Gesture Recognition in the Bolivian Sign Language, using computer vision to enable machines in the interpretation thus help close the breach on communication with people having issues to communicate verbally.


## Dataset


Since this is the first approach ever made on dynamic gesture concerning LSB, we had to make our own dataset. To be able to download the dataset LSA12 folow this link, this dataset consist on twelve gestures commonly used and were selected alongside interpreters.
1. Ayuda
2. Bolivia
3. Como
4. Dinero
5. Doctor
6. Donde
7. Explicacion
8. Guardar
9. Necesito
10. Quien(Occidente)
11. Quien(Oriente)
12. Saludos

Some features of the proccesss that might be relevant for future work, the height of the camera POV was 1.5 meters and the length towards the intepreter was 2 meters long, using a black background.

## Feature Extracction

We worked on this project using Mediapipe, using some of the points that can be extracted via it's Pose Estmiation module. The algorithm that exectues the iterarionts for each frame in each video is on prepareData.py

## Trainning

The RNN was made using Tensorflow+Keras, we had the best perfomance using Bidirecctional NN with Recurrent dropouts. As a suggesttion for those who attemp the same method try playing with the temporal window, the best perfomance comes aorund 75 to 90 frames

## GUI

As a last step for this project, the presentation for the user comes as as GUI that enables to control the source(camera/video) as well as the txt2sound feature. This can be improved because it uses PyQT5 threads.
