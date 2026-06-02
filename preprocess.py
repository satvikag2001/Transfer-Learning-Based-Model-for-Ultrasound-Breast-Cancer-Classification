import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
model = tf.keras.models.load_model("vgg19-90.6.h5")
counter =0
total = 0
images = os.listdir("BreaKHis 400X/BreaKHis 400X/BreaKHis 400X/test/benign")
total+=len(images)
for i in images:
  
  image = load_img('BreaKHis 400X/BreaKHis 400X/BreaKHis 400X/test/benign/'+i, target_size=(224, 224, 3))
  #np.expand_dims(image, axis=0)
 
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = image/255.
  
  #image = preprocess_input(image)
  if model.predict(image)<0.5 :
    counter+=1 

images = os.listdir("BreaKHis 400X/BreaKHis 400X/BreaKHis 400X/test/malignant")
total+=len(images)
for i in images:
  
  image = load_img('BreaKHis 400X/BreaKHis 400X/BreaKHis 400X/test/malignant/'+i, target_size=(224, 224, 3))
  #np.expand_dims(image, axis=0)
 
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = image/255.
  
  #image = preprocess_input(image)
  if model.predict(image)>0.5:
    counter+=1 
print(counter/total)  
