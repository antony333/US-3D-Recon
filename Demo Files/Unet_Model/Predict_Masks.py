"""
@author: Antony
"""


from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf



#Following varibles are passed from Matlab Environment
#unet_location
#total frames 
#s_frame
#e_frame

#%%

#Loading Unet Model
filename_unet = unet_location+'/Unet_Model/Original_Unet32.h5'
unet_model =  load_model(filename_unet, compile=False)
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#%%

#To load the input us image data
Dataset = np.zeros((256,128,totalframes))
Predicted_Masks = []
for i in range(totalframes):
    filename = unet_location + '/US Images/ImgAHist_'+str(i+1)+'.png'
    Dataset[:,:,i] = cv2.imread(filename,0)

#To normalize the input image data
Dataset = np.array(Dataset)/255.
Req_Dataset = Dataset[:,:,s_frame:e_frame]



#To predict masks
for i in range(np.shape(Req_Dataset)[2]):
  test_img_input=np.expand_dims(Req_Dataset[:,:,i], 0)
  prediction = (unet_model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
  Predicted_Masks.append(prediction)

Predicted_Masks = np.array(Predicted_Masks)
