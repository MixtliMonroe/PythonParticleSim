import numpy as np

class Particle():
  def __init__(self, position = None, velocity = None, radius = None):
    if position is None:
      self.position = np.array([0., 0., 0.])
    else:
      self.position = position
    
    if velocity is None:
      self.velocity = np.array([0., 0., 0.])
    else:
      self.velocity = velocity
    
    if radius is None:
      self.radius = 0.
    else:
      self.radius = radius
  
  def set_position(self, x):
    self.position = x

  def change_position(self, dx):
    self.position += dx

  def set_velocity(self, v):
    self.velocity = v

  def change_velocity(self, dv):
    self.velocity += dv

  def set_radius(self, r):
    self.radius = r

  def change_radius(self, dr):
    self.radius += dr
