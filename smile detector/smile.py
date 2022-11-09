import cv2

#set the respective size for the pop-up-window
# cv2.namedWindow('Hello', cv2.WINDOW_NORMAL)  

#which generate the random values for the color sequence
from random import randrange

#importing the algorithm and save to the custome variable 
#front face algorithm
face_classifier=cv2.CascadeClassifier('face_algorithm.xml')
#face in smile algorithm
smile_classifier=cv2.CascadeClassifier('smile_algorithm.xml')

#adding the video or the live-stream to the variable
livefeed=cv2.VideoCapture('smile.mp4')

#which run the loop to the infinite times
while True:

    #current frame 
    successful_vidobject_read, vid_object = livefeed.read()

    #which converts the color file or live stream to the gray scale or black and white
    grayscale_livefeed = cv2.cvtColor(vid_object, cv2.COLOR_BGR2GRAY)

    #find the coordinates of the frame
    face_coordinates = face_classifier.detectMultiScale(grayscale_livefeed)

    #loop to detect the face using the coordinates
    for (x,y,w,h) in face_coordinates:

        #draw the rectangle around the face
        cv2.rectangle(vid_object, (x,y), (x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)
        
        #extract only the face image from the whole image 
        sub_image=vid_object[y:y+h, x:x+w]

        #convert the sub image to the gray scale
        sub_grayscale = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        
        #detect the smile in the extracted face
        smile_coordinates = smile_classifier.detectMultiScale(sub_grayscale, scaleFactor=1.7, minNeighbors=20)
        
        # for (x_,y_,w_,h_) in smile_coordinates:
        #     cv2.rectangle(sub_image, (x_,y_), (x_+w_,y_+h_),(255,255,256),4)

        #print the text as smiling below the rectangle of the face 
        if len(smile_coordinates) > 0:
            cv2.putText(vid_object, 'smiling', (x, y+h+40), fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(randrange(300),randrange(300),randrange(300)))

    #which shows the image in the pop up window
    vid_object = cv2.resize(vid_object, (500, 500))
    cv2.imshow('Smile Detector' , vid_object)
    
    #which stops the window
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

#distructer of the whole file 
livefeed.release()
cv2.destroyAllWindows()

print("process completed")