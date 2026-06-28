import numpy as np

class Particle():
  def __init__(self, position = None, velocity = None, mass = None, radius = None):
    if position is None:
      self.position = np.array([0., 0., 0.])
    else:
      self.position = position
    
    if velocity is None:
      self.velocity = np.array([0., 0., 0.])
    else:
      self.velocity = velocity

    if mass is None:
      self.mass = 0.
    else:
      self.mass = mass

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

  def set_mass(self, m):
    self.mass = m

  def set_radius(self, r):
    self.radius = r

class ParticleSim():
  def __init__(self):
    self.particles = []
    self.gravity = np.array([0., 0., 0.])
    self.cor = 1
    self.time = 0

  def add_particles(self, particles):
    assert type(particles) == list, "Appended particle(s) must be in a list"

    self.particles.extend(particles)

  def set_gravity(self, gravity):
    self.gravity = gravity
  
  def set_cor(self, cor):
    self.cor = cor

  def update_state(self, dt=1e-3):
    for P in self.particles:
      P.change_position(P.velocity * dt + self.gravity/2 * dt**2)
      P.change_velocity(self.gravity * dt)
    self._collide_particles_pairwise()
    self._boundary_conditions()

    self.time += dt

  def _collide_particles_pairwise(self):
    '''
    Check which pair of particles are colliding, and apply collision when approrpriate
    '''

    for i in range(len(self.particles)):
      for j in range(i+1, len(self.particles)):
        particle1 = self.particles[i]
        particle2 = self.particles[j]
        if sum((particle1.position - particle2.position)**2) <= (particle1.radius + particle2.radius)**2:
          self._collide_pair(particle1, particle2)
  
  def _collide_pair(self, particle1, particle2):
    # Calculate the normal impulse
    reduced_mass = particle1.mass * particle2.mass/(particle1.mass + particle2.mass)
    collision_normal  = (particle2.position - particle1.position) / np.linalg.norm(particle2.position - particle1.position) # Normalised
    relative_velocity =  particle2.velocity - particle1.velocity
    normal_impulse = reduced_mass * (1 + self.cor) * np.dot(relative_velocity, collision_normal)

    # Update velocities
    deltaV1 =  (normal_impulse/particle1.mass) * collision_normal
    deltaV2 = -(normal_impulse/particle2.mass) * collision_normal
    particle1.change_velocity(deltaV1)
    particle2.change_velocity(deltaV2)

    # Snap particles apart so that the numerics don't cause them to clip
    # Keeping the centre of mass constant
    total_mass = particle1.mass + particle2.mass
    sum_radii  = particle1.radius + particle2.radius
    centre_of_mass = (particle1.mass * particle1.position + particle2.mass * particle2.position)/total_mass
    particle1.set_position(centre_of_mass - sum_radii * particle2.mass/total_mass)
    particle2.set_position(centre_of_mass + sum_radii * particle1.mass/total_mass)
  
  def _boundary_conditions(self, width = 10, height = 10, depth = 10):

    dims = [width, height, depth]

    for P in self.particles:
      # Along each axis
      for i in range(3):

        # Bounce off wall at x,y,z = 0
        if P.position[i] - P.radius < 0:
          P.position[i] = P.radius
          P.velocity[i] *= -1

        # Bounce off wall at x,y,z = width,height,depth
        elif P.position[i] - P.radius > dims[i]:
          P.position = dims[i] - P.radius
          P.velocity[i] *= -1