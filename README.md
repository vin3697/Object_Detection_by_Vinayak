# Object_Detection_by_Vinayak

#Download the weight file from officail webiste of YOLOV3 algorithm
I have used Transfer Learning in this project, where I have taken the pre-trained model(YOLOV3) weights and biases.

The conifgration and weight files has been loaded in my main code so that it classifies the objects which are being detected by my WebCam.

It captures the frame and detects the object depending upon the weights of pretrained DNN model and at Ouput layer it gives us the probability.
We capture the label which has the hightest probability and that label is displayed on the frame window.


I have tried the object detection with YOLOV3 tiny as well as with YOLOV4 tiny, I observed that the result is less accurate but the speed has increased.
On CPU YOLov4 tiny is pretty much fast compared to YOLOv3 tiny!

To do the detection with yolov3-tiny and yolov4-tiny just download the weigth and config file from the webiste.
And you have to change the file names in the code as well, in code on line 5 - we have to give the dnn network the yolo version which you want to use!!

##the below pic has used yolo4 tiny model for object detection
![tolo4-tiny](https://user-images.githubusercontent.com/92587549/139331930-bcf5d398-8ee6-4c68-9296-f773b10099d0.jpeg)
