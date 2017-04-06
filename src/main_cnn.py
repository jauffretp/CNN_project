# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:13:17 2016

@author: Pascal
"""



import sys, argparse,glob
import numpy as np
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
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
	args = parser.parse_args()

	
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	json_file = open('cnn_point_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("cnn_point.h5")
	




	cap = cv2.VideoCapture(0)
	
	while True:
		ret,array_picture_bgr = cap.read()
		


		gray = cv2.cvtColor(array_picture_bgr, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		faces_dict = []
		for (x,y,w,h) in faces:

			width_factor = float(w)/96.
			height_factor = float(h)/96.

			face_img = gray[y:y+h, x:x+w].copy()
			face_img = cv2.resize(face_img, (96, 96))


			face_img = face_img.reshape(1,96,96,1)
			preds = loaded_model.predict(face_img)[0]
			
			obj = {}
			obj["preds"] = preds
			obj["x"] = x
			obj["y"] = y
			obj["x_factor"] = width_factor
			obj["y_factor"] = height_factor
			obj["face_img"] = face_img
			faces_dict.append(obj)

		display_picture(array_picture_bgr,faces_dict, faces)

		key = cv2.waitKey(1)
			


def  display_picture(array_picture_bgr, faces_dict, faces):
		
	for face_dict in faces_dict:
		preds = face_dict["preds"]
		x = face_dict["x"]
		y = face_dict["y"]

		face_img = face_dict["face_img"][0]
		


	

		x_factor = face_dict["x_factor"]
		y_factor = face_dict["y_factor"]

		for i in range(0,len(preds), 2):
			x_point = preds[i]
			y_point = preds[i+1]

			''' 
			print("x: ",x)
			print("x point ",x_point)
			print("x factor : ",x_factor)
			print("final x : ", x + (x_point*x_factor))
			'''


			y_final = int(y + (y_point*y_factor))
			x_final = int(x + (x_point*x_factor))
			cv2.circle(array_picture_bgr, (x_final,y_final), radius=1, color=(255,0,0), thickness=3)
			cv2.circle(face_img, (x_point,y_point), radius=1, color=(255,0,0), thickness=3)

		cv2.imshow('face',face_img)
	


	
	for (x,y,w,h) in faces:
		cv2.rectangle(array_picture_bgr,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow('CNN Detection',array_picture_bgr)
	


if __name__ == "__main__":
	main(sys.argv)