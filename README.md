# face_rec
A neural network trained to classify images of family members

Convolutional neural network architecture comprises of 3 convolutional layers followed by 2 fully connected linear layers. 
Network trained on limited custom dataset created small image set expanded through contrast adjustments. 

train.py : trains the network; also contains the pref() function to use trained network to infer on validation set

build_ds.py : the script to increase dataset size by apply PIL filters to the original images

dataset.py : contains the implementation of the dataset class
