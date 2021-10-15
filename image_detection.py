import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()


#print(classes)

capture = cv2.VideoCapture(0)

while True:

    ret, frame = capture.read()

    cv2.imshow('Frame', frame)
    height , width, _= frame.shape

    #swapRB is used to swap from RGB to BGR 
    blob = cv2.dnn.blobFromImage(frame, 1/255 , (416, 416), (0,0,0), swapRB=True , crop=False) #blob is used so that we can give the image as an input to the DNN into three channels(RGB)

    net.setInput(blob) #This sets blob as an input to the DNN netwrok
    output_layers_names = net.getUnconnectedOutLayersNames() #to get the names at output layers (labels-softmax function)
    layerOutputs = net.forward(output_layers_names)

    boxes =[]
    confidences=[]
    class_ids=[]

    #we need 2 for loops as we have to loop over the output layers
    #the 1st for loop is used to extract info. from output layers
    #the 2nd loop is used to extract info. which is contained inside the output layers

    for output in layerOutputs:
        for detection in output:
            #detection contains 4 parameters of bounding box, one for confidence , and one for class_id which gives us 80 classes with probabilities
            #so we have 4+1+1+80= 86 parameters [0,1,2....85]
            scores = detection[5:] # this stores all the probabilities of different classes ids
            class_id = np.argmax(scores) #this command gives us the hightest probability
            confidence = scores[class_id]
            
            #now if probability is greater than 0.5 we are going to draw a bounding box around our object

            if confidence > 0.5:
                center_x = int (detection[0]*width) # detection[] gives us the normalized image size so to convert to original we 
                center_y = int (detection[1]*height)
                w = int (detection[2]*width)
                h = int (detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2) # this gives the positions of the upper left corner of image

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    #print(len(boxes))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes.flatten())


    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes) ,3)) #3 is for RGB channels

    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color =colors[i]
        cv2.rectangle(frame, (x,y), (x+w , y+h), color, 2)
        cv2.putText(frame, label+ " "+ confidence, (x,y+20), font, 2, (255,255,255), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
frame.release()
cv2.destroyAllWindows()