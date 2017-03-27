import numpy as np
from numpy.random import *
from random import randint
from scipy.special import ndtri
from skimage import color
from skimage import feature
from skimage import io
from skimage import segmentation
from skimage import draw
import scipy
from scipy import misc




def multinomial_resample(weights):
    """

   Parameters
   ----------

    weights : list-like of float
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes
        
    """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid error, sum=1
    return np.searchsorted(cumulative_sum, random(len(weights)))




def update_particles(particles,std=1):
    """

   Parameters
   ----------

    particles : particles dictionnary
                key x : array of x-coordinates of particles
                key y : array of y-coordinates of particles
    
    std : standard deviation for normal distribution
    Returns
    -------

    updated particles based on normal random step
    """
    particles['x']=particles["x"]+np.random.normal(loc=0.0, scale=std, size=len(particles["x"]))
    particles['y']=particles["y"]+np.random.normal(loc=0.0, scale=std, size=len(particles["y"]))
    return particles
    

def multinomial_resample_mk2(weights):
    """

   Parameters
   ----------

    weights : list-like of float
        list of weights as floats

   Returns
    -------

    indexes : ndarray of ints
        array of indexes
        
    """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid error, sum=1
    rand=np.sort(random(len(weights)))
    #print(rand)
    #print(weights)
    return np.searchsorted(cumulative_sum, rand)    
    
    
def update_particles_mk2(particles,center):
    """
    WORK IN PROGRESS
   Parameters
   ----------

    particles : particles dictionnary
                key x : array of x-coordinates of particles
                key y : array of y-coordinates of particles
    centers : 
    
    Returns
    -------

    updated particles based on motion equation
    """
    
    particles['x']=particles["x"]+np.random.normal(loc=0.0, scale=1.0, size=len(particles["x"]))
    particles['y']=particles["y"]+np.random.normal(loc=0.0, scale=1.0, size=len(particles["y"]))
    return particles
    

'''
def update_particles_v2(particles, std=1):
    particles['x'] = np.random.normal(loc=particles['x'], scale=std)
'''

""" 
#For instance (integration purpose)
index=multinomial_resample(particles[weights])
aux_particles={}
aux_particles['x']=[ particles['x'].[i] for i in index]
aux_particles['y']=[ particles['y'].[i] for i in index]
new_particles=udpate_particles(aux_particles)
"""