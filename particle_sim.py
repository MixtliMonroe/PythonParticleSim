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
    self.cor  = 1
    self.time = 0
    self.dims = [10, 10, 10]

  def add_particles(self, particles):
    assert type(particles) == list, "Appended particle(s) must be in a list"

    self.particles.extend(particles)
    self._collide_particles_pairwise()

  def set_gravity(self, gravity):
    self.gravity = gravity
  
  def set_cor(self, cor):
    self.cor = cor

  def set_boundaries(self, width, height, depth):
    self.dims = [width, height, depth]

  def update_state(self, dt=1e-3):
    for P in self.particles:
      P.change_position(P.velocity * dt + self.gravity/2 * dt**2)
      P.change_velocity(self.gravity * dt)
    self._boundary_conditions()
    self._collide_particles_pairwise()

    self.time += dt

  def _collide_particles_pairwise(self):
    '''
    Check which pair of particles are colliding, and apply collision when appropriate
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
    particle1.set_position(centre_of_mass - collision_normal * sum_radii * particle2.mass/total_mass)
    particle2.set_position(centre_of_mass + collision_normal * sum_radii * particle1.mass/total_mass)
  
  def _boundary_conditions(self):

    for particle in self.particles:
      # Along each x,y,z axis
      for i in range(3):

        # Bounce off wall at x,y,z = 0
        if particle.position[i] - particle.radius < 0:
          particle.position[i] = particle.radius
          particle.velocity[i] *= -self.cor

        # Bounce off wall at x,y,z = width,height,depth
        elif particle.position[i] + particle.radius > self.dims[i]:
          particle.position[i] = self.dims[i] - particle.radius
          particle.velocity[i] *= -self.cor