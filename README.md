# Object_Detection_by_Vinayak

#Download the weight file from officail webiste of YOLOV3 algorithm
I have used Transfer Learning in this project, where I have taken the pre-trained model(YOLOV3) weights and biases.

The conifgration and weight files has been loaded in my main code so that it classifies the objects which are being detected by my WebCam.

It captures the frame and detects the object depending upon the weights of pretrained DNN model and at Ouput layer it gives us the probability.
We capture the label which has the hightest probability and that label is displayed on the frame window.
