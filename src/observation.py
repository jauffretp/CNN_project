# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:18:03 2016

@author: Alexis
"""

import numpy as np
from skimage import color
from skimage import feature
from skimage import io
from skimage import segmentation
from skimage import draw

threshold = 0.05

def evaluate(image_array, particles):
    """ Calcule pour chaque particle son likelihood et son meilleur 
    cercle. 
    Temps d'execution : 473 ms
    
    Args:
    	-image array : matrice numpy de dimension 3 (axes 1, 2 : axes x, y 
     de l'image, axes 3 : R,G,B)
       -particles : vecteur de dictionnaires {x:, y:, weight:}
    	
    Returns:
    	likelihood...
    """
    
    # On converti l'image au format YCbCr
    image_array_YCbCr = rgb2ycbcr(image_array)
    
    # On construit aussi l'image en likelihood
    image_likelihood = skin_likelihood(image_array_YCbCr)
    
    # On ajoute pour chaque particule la valeur du likelihood
    for particle in particles:
        particle['skin_likelihood'] = image_likelihood[particle['x'],particle['y']]

    # On calcule les cercles
    particles = generate_circles(particles)
        
    # on calcule les likelihood des cercle
    particles = cercle_likelihood(image_array, image_likelihood, particles)
    
    # on calcule les likelihood finaux
    lambda1 = 1
    lambda2 = 1
    global threshold
    for particle in particles:
        if particle['skin_likelihood'] < threshold:
            particle['likelihood'] = 0
        else:
            particle['likelihood'] = lambda1*particle['skin_likelihood'] + lambda2*particle['best_cercle']['cercle_likelihood']
    
    return particles 

def rgb2ycbcr(image_RGB):
    """ Transforme une image au format RGB en une image au format YCbCr
    selon la méthode décrite dans l'article. Méthode absente de la 
    version stable de skimage, mais déjà terminée en dev.
    temps d'execution : 100 ms 
    
    Args:
    	matrice numpy de dimension 3 (axes 1, 2 : axes x, y de l'image
     axes 3 : R,G,B)
    	
    Returns:
    	matrice numpy de dimension 3 (axes 1, 2 : axes x, y de l'image
     axes 3 : Y,Cb,Cr)
    """
    
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image_RGB.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
    
def skin_likelihood(image_ycbcr):
    """ Calcule pour chaque pixel de l'image la valeur du likelihood
    qui représente la probabilité que le pixel soit de la peau. 
    La distribution utilisée est une loi normale bi-dimensionnelle, où
    la moyenne et la matrice de variance covariance est fixée (valeurs 
    classique qui fonctionnent bien)
    Temps d'execution : 150 ms
    
    Args:
    	matrice numpy de dimension 3 (axe 1,2 : coordonnées x,y, 
     axe 3 : Y, Cb, Cr)
    	
    Returns:
    	matrice numpy de dimension 2 (axe 1,2 : coordonnées x,y)
    """
    m = np.matrix([120,152])
    C = np.matrix([[85,-55],
                   [-55,85]]) # a modifier plus tard avec les vrais valeurs
    C_inv = np.linalg.inv(C)
    
    image_cbcr = image_ycbcr[:,:,1:]
    image_cbcr = image_cbcr - np.array([120,150])
    s1=np.einsum('ijk,kl->ijk',image_cbcr,C_inv)
    s2=np.einsum('ijk,ijk->ij',image_cbcr,s1)
    return np.exp(-s2/2)
    
    
def cercle_likelihood(image, image_likelihood, particles):
    """ Calcule pour chaque cercle candidat son likelihood
    Temps d'execution : 250 ms
    
    Args:
     image originale : matrice numpy de dimension 3 (axe 1,2 : coordonnées x,y, 
     axe 3 : R, G, B)
     image 'version' skin likelihood  : matrice numpy de dimension 2
     particles : numpy array de dimension 1 (axe 1 : dictionnaire)
    	
    Returns:
    	particles : numpy array de dimension 1 (axe 1 : dictionnaire)
    """
    
    # on construit l'image transformée en contour
    image_grey = color.rgb2grey(image) # 33 ms
    image_contour = feature.canny(image_grey) # 140 ms
    
    # On construit ensuite l'image des likelihood    
    global threshold
    image_likelihood_bin = (image_likelihood>threshold).astype(np.uint8) # 1.48 ms
    
    # Pour gagner du temps on calcule ici le premier AND
    image_AND = image_contour*image_likelihood_bin
    
    # on constuit enfin chaque cercle et on enregistre son likelihood associé
    # enfin, on conserve le meilleur cercle
    for particle in particles:
        ind_best_cercle = 1
        lik_best_cercle = 0
        ind = -1
        for cercle in particle['cercles']:
            ind = ind+1
            image_cercle = np.zeros(shape=image_grey.shape, dtype=np.uint8)
            rr, cc = draw.circle_perimeter(cercle['r'], cercle['c'], cercle['radius'])
            
            index1 = np.argwhere(rr<0)
            index2 = np.argwhere(cc<0)
            index3 = np.argwhere(rr>=image_grey.shape[0])
            index4 = np.argwhere(cc>=image_grey.shape[1]) 
            index = np.concatenate((index1,index2,index3,index4),axis=0)
            rr = np.delete(rr, index)
            cc = np.delete(cc, index)
            
            image_cercle[rr,cc] = 1
            
            # Et on calcule le likelihood du cercle
            image_AND_cercle = image_AND*image_cercle
            cercle['cercle_likelihood'] = np.sum(image_AND_cercle)/np.sum(image_cercle)
        
            # On enregistre la nouvelle valeur si elle est mieux
            if cercle['cercle_likelihood'] > lik_best_cercle:
                ind_best_cercle = ind
                lik_best_cercle = cercle['cercle_likelihood']

        #Si c'est une particle qui a des cercles, on enregistre le meilleur cercle proposé 
        if(particle['skin_likelihood']>threshold):
            particle['best_cercle'] = particle['cercles'][ind_best_cercle]
    
    return particles
        

def generate_circles(particles):
    """ Génère aléatoirement des cercles centrés sur chaque particule,
    mais de rayon tiré uniformément dans l'intervalle 10 pixels - 200 pixels
    Temps d'execution : 468 µs
    
    Args:
    	matrice numpy de dimension 1 (dictionnaire)
    	
    Returns:
    	matrice numpy de dimension 1 (dictionnaire)
    """
    
    # skin color threshold value
    global threshold
    
    # On conserve les particles qui ont un likelihood > threshold
    particles_kept = np.array([particle for particle in particles if particle['skin_likelihood']>threshold])
    
    # Nombre de cercles générées au hasard
    N_cercles = 5
    
    for particle in particles:
        # On génère des ellipses candidates que pour les particles suivantes
        if particle['skin_likelihood'] > threshold:
            # On crée les objets ellipses
            cercles = np.array([{'r': particle['x'], 
                                 'c': particle['y'], 
                                 'radius': np.random.randint(low=10.0, high=200.0, size=1)[0], 
                                 } for i in range(N_cercles)
                                 ])
            particle['cercles'] = cercles
        else:
            particle['cercles'] = np.array([])

    return particles
    
"""
#image = io.imread("..\\..\\scarlett.jpeg")    
image = io.imread("..\\data\\sequence1\\sequence10047.png")   
    
N_particles = 500

particles = np.array([{'x': np.random.randint(low=0,high=image.shape[0]),
                       'y': np.random.randint(low=0,high=image.shape[1])
                       } for i in range(N_particles)
                      ])
particles = evaluate(image,particles)   

image_ycbcr = rgb2ycbcr(image)
image_likelihood = skin_likelihood(image_ycbcr)
image_skin = (image_likelihood>threshold).astype(np.uint8)
io.imshow(image_skin)"""