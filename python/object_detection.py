# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:13:17 2016

@author: Pascal
"""



import sys, argparse,glob
import numpy as np
from keras.applications.vgg19 import VGG19,preprocess_input, decode_predictions

from keras.preprocessing import image
from keras.models import model_from_json

import cv2



"""
Execute main script

Exemple of execution :
python3 main.py --source webcam
python3 main.py --source "random"

"""


def main(argv):
	parser = argparse.ArgumentParser(description='Main script')
	parser.add_argument('--source', required=True, help='source for the pictures')
	args = parser.parse_args()
	source = args.source

	#model = ResNet50(weights='imagenet')
	model = VGG19(weights='imagenet')

	#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


	

	if(source != "webcam"):
		file_list = glob.glob("../data/" + source + "/*")
		print(file_list)
		
		for file_picture in file_list:
			print(file_picture)
			img = image.load_img(file_picture, target_size=(224, 224))



			features = image.img_to_array(img)
			features = np.expand_dims(features, axis=0)
			features = preprocess_input(features)
			preds = model.predict(features)
			preds = decode_predictions(preds)


			original_picture = cv2.imread(file_picture,1)
			display_picture(original_picture, preds)


			
			key = cv2.waitKey(20000)





	else:
		

		cap = cv2.VideoCapture(0)
		
		while True:
			ret,array_picture_bgr = cap.read()
			

			features = cv2.resize(array_picture_bgr, (224, 224)) 
			features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
			features = features.astype('float64')

			
			features = np.expand_dims(features, axis=0)
			features = preprocess_input(features)


			preds = model.predict(features)

			preds = decode_predictions(preds)

			#print('Predicted:', preds[0][0])
			#print("\n")

			#print(preds[0])
			gray = cv2.cvtColor(array_picture_bgr, cv2.COLOR_BGR2GRAY)
			#faces = face_cascade.detectMultiScale(gray, 1.3, 5)



			display_picture(array_picture_bgr, preds)

			key = cv2.waitKey(1)
			


def  display_picture(array_picture_bgr, preds):
	format_string = [pred[1] + " : " + str(pred[2]) for pred in preds[0]]

	y_ini = 30
	y_offset = 30
	for idx, string in enumerate(format_string):
		y = y_ini + y_offset*idx
		cv2.putText(array_picture_bgr, format_string[idx], (10, y), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 2)
	
	'''
	for (x,y,w,h) in faces:
		cv2.rectangle(array_picture_bgr,(x,y),(x+w,y+h),(255,0,0),2)
	'''

	cv2.imshow('CNN Detection',array_picture_bgr)


	



if __name__ == "__main__":
	main(sys.argv)