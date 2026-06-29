import numpy as np
import pygame
from particle_sim import Particle, ParticleSim

# Initialize pygame
pygame.init()

# Simulation parameters
width, height = 800, 600
FPS = 60
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()

# Create simulation
Simulation = ParticleSim()
Simulation.set_boundaries(width=width/10, height=height/10, depth=1)
Simulation.set_gravity(np.array([0., 10., 0.]))
Simulation.add_particles([Particle(position=np.array([np.random.rand()*width/10, np.random.rand()*height/10, .5]),
                                    velocity=np.array([20*np.random.rand()-10, 20*np.random.rand()-10, 0.]),
                                    mass=1 + 2*np.random.rand(),
                                    radius=1) for _ in range(100)])

running = True
while running:
  # Handle events
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
  
  # Update simulation
  Simulation.update_state(dt=1/FPS)
  
  # Render
  screen.fill((255, 255, 255))
  
  for particle in Simulation.particles:
    # Draw particle (project 3D position to 2D, using x-y plane)
    x = int(particle.position[0] * 10)
    y = int(particle.position[1] * 10)
    radius = int(particle.radius * 10)
    pygame.draw.circle(screen, (0, 0, (1-particle.mass/3)*255), (x, y), radius)
  
  pygame.display.flip()
  clock.tick(FPS)

pygame.quit()