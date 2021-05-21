import random
import os
import numpy as np
from itertools import product
from espressomd.io.writer import vtf
from espressomd.virtual_sites import VirtualSitesRelative
from espressomd import lb
from espressomd import interactions
import espressomd
espressomd.assert_features(["ROTATION", "ROTATIONAL_INERTIA", "EXTERNAL_FORCES",
                            "MASS", "VIRTUAL_SITES_RELATIVE", "LENNARD_JONES"])


# System parameters
#############################################################
box_l = 40.  # size of the simulation box

skin = 3  # Skin parameter for the Verlet lists
time_step = 0.001
eq_tstep = 0.001

n_cycle = 1000  # 100
integ_steps = 150

# Interaction parameters (Lennard-Jones for raspberry)
#############################################################

# the subscript c is for colloid and s is for salt (also used for the surface beads)
agrid = 1
eps_ss = 1   # LJ epsilon between the colloid's surface particles.
sig_ss = 1   # LJ sigma between the colloid's surface particles.
eps_cs = 48.  # LJ epsilon between the colloid's central particle and surface particles.
radius_col = 3
harmonic_radius = radius_col
# LJ sigma between the colloid's central particle and surface particles (colloid's radius).
sig_cs = radius_col

# System setup
#############################################################
system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = skin
system.periodicity = [True, True, True]
# the LJ potential with the central bead keeps all the beads from simply collapsing into the center
system.non_bonded_inter[1, 0].wca.set_params(epsilon=eps_cs, sigma=sig_cs)
# the LJ potential (WCA potential) between surface beads causes them to be roughly equidistant on the
# colloid surface
system.non_bonded_inter[1, 1].wca.set_params(epsilon=eps_ss, sigma=sig_ss)

# the harmonic potential pulls surface beads towards the central colloid bead
col_center_surface_bond = interactions.HarmonicBond(k=3000., r_0=harmonic_radius)
system.bonded_inter.add(col_center_surface_bond)

# for the warmup we use a Langevin thermostat with an extremely low temperature and high friction coefficient
# such that the trajectories roughly follow the gradient of the potential while not accelerating too much
random.seed(os.getpid())
seed_pass = random.choice(range(os.getpid()))
print(seed_pass)
system.thermostat.set_langevin(kT=0.00001, gamma=40., seed=seed_pass)

print("# Creating raspberry")
center = system.box_l / 2
colPos = (0, 0, 0)

# Number of particles making up the raspberry (surface particles + the central particle). use liberally
n_col_part = round(4*np.pi*pow(radius_col, 2)/pow(agrid, 2))+5
n_fill_part = round(4*np.pi*pow(radius_col-sig_ss/2., 3)/(3*pow(agrid, 3)))+5
print(n_col_part)
print(n_fill_part)
# Place the central particle
system.part.add(id=0, pos=colPos, type=0, fix=(True, True, True),
                rotation=(1, 1, 1))  # Create central particle

# Create surface beads uniformly distributed over the surface of the central particle
for i in range(1, n_col_part+1):
    colSurfPos = np.random.randn(3)
    colSurfPos = colSurfPos / np.linalg.norm(colSurfPos) * radius_col + colPos
    system.part.add(id=i, pos=colSurfPos, type=1)
    system.part[i].add_bond((col_center_surface_bond, 0))
print("# Number of colloid beads = {}".format(n_col_part))

# Relax bead positions. The LJ potential with the central bead combined with the
# harmonic bond keep the monomers roughly radius_col away from the central bead. The LJ
# between the surface beads cause them to distribute more or less evenly on the surface.
system.force_cap = 1000
system.time_step = eq_tstep

print("Relaxation of the raspberry surface particles")
for i in range(n_cycle):
    system.integrator.run(integ_steps)

# Restore time step
system.time_step = time_step

# this loop moves the surface beads such that they are once again exactly radius_col away from the center
# For the scalar distance, we use system.distance() which considers periodic boundaries
# and the minimum image convention
colPos = system.part[0].pos
for p in system.part[1:]:
    p.pos = (p.pos - colPos) / np.linalg.norm(system.distance(p, system.part[0])) * radius_col + colPos
    p.pos = (p.pos - colPos) / np.linalg.norm(p.pos - colPos) * radius_col + colPos


# Select the desired implementation for virtual sites
system.virtual_sites = VirtualSitesRelative()
# Setting min_global_cut is necessary when there is no interaction defined with a range larger than
# the colloid such that the virtual particles are able to communicate their forces to the real particle
# at the center of the colloid
system.min_global_cut = radius_col

# Calculate the center of mass position (com) and the moment of inertia (momI) of the colloid
com = np.average(system.part[1:].pos, 0)  # system.part[:].pos returns an n-by-3 array
momI = 0
for i in range(n_col_part):
    momI += np.power(np.linalg.norm(com - system.part[i].pos), 2)

# note that the real particle must be at the center of mass of the colloid because of the integrator
print("\n# moving central particle from {} to {}".format(system.part[0].pos, com))
system.part[0].fix = [False, False, False]
system.part[0].pos = com
system.part[0].mass = n_col_part
print(momI)
system.part[0].rinertia = np.ones(3) * momI

# Convert the surface particles to virtual sites related to the central particle
# The id of the central particles is 0, the ids of the surface particles start at 1.
for p in system.part[1:]:
    p.vs_auto_relate_to(0)

system.part[0].pos = (0, 0, 0)
system.integrator.run(1)

# the LJ potential with the central bead keeps all the beads from simply collapsing into the center
# system.non_bonded_inter[2, 0].wca.set_params(epsilon=eps_cs, sigma=sig_cs)
# the LJ potential (WCA potential) between surface beads causes them to be roughly equidistant on the
# colloid surface
system.non_bonded_inter[2, 1].wca.set_params(epsilon=eps_ss, sigma=sig_ss)
system.non_bonded_inter[2, 2].wca.set_params(epsilon=eps_ss, sigma=sig_ss)
novi_kolacici = max(system.part[:].id)+1

delta_R = radius_col-sig_ss/2
num_of_bins = int(np.ceil(delta_R))
dist_from_center = [0 for _ in range(num_of_bins)]

for j in range(novi_kolacici, novi_kolacici+n_fill_part+1):
    bn = int(np.ceil(random.random()*delta_R))
    while dist_from_center[bn-1] > 4*np.pi*(bn + 1)**2:
        bn = int(np.ceil(random.random()*delta_R))
    dist_from_center[bn-1] += 1


occupied = [(0, 0, 0)]
for index, i in enumerate(range(novi_kolacici, novi_kolacici+n_fill_part+1)):

    bn = 0
    for id, b in enumerate(dist_from_center):
        if b > 0:
            bn = id + 1
            dist_from_center[id] -= 1
            break

    colSurfPos = np.random.randn(3)
    colSurfPos = colSurfPos / np.linalg.norm(colSurfPos)*delta_R*bn/num_of_bins + colPos
    distances = [np.linalg.norm(colSurfPos-i) for i in occupied]

    while not np.all(i >= 1 for i in distances):
        colSurfPos = np.random.randn(3)
        colSurfPos = colSurfPos / np.linalg.norm(colSurfPos)*delta_R*bn/num_of_bins + colPos
        distances = [np.linalg.norm(colSurfPos-i) for i in occupied]

    occupied.append(colSurfPos)
    system.part.add(id=i, pos=colSurfPos, type=2)
print("# Number of colloid beads = {}".format(n_fill_part))

system.force_cap = 100
system.time_step = eq_tstep
print("Relaxation of the raspberry surface particles")

for i in range(n_cycle):
    system.integrator.run(int(integ_steps))
    # system.galilei.kill_particle_motion()
    system.force_cap = 1000
system.time_step = time_step

for p in system.part[novi_kolacici:]:
    p.vs_auto_relate_to(0)

# make sure that the mean values bellow correspond to values you expect!!!
# also by tuning the nuber of partciles in the raspberry try to minimise the st_dev! you will see once you cant get better, doesnt take much time
slice_master = [x.id for x in system.part[:] if x.type == 1]
lister = [np.linalg.norm(system.part[i].pos - system.part[j].pos)
          for i, j in product(slice_master, slice_master) if i != j]
lister.sort(reverse=False)
print('mean of shell: ', np.mean(lister[:n_col_part-10]))
print('std dev of shell:', np.std(lister[:n_col_part-10]))

slice_master = [x.id for x in system.part[:] if x.type == 2]
lister = [np.linalg.norm(system.part[i].pos - system.part[j].pos)
          for i, j in product(slice_master, slice_master) if i != j]
lister.sort(reverse=False)
print('mean of core: ', np.mean(lister[:n_col_part-10]))
print('std dev of core:', np.std(lister[:n_col_part-10]))

system.part[0].pos = (0, 0, 0)
system.integrator.run(1)
custom_data = open('raspberry_visual.vtf', mode='w+t')
vtf.writevsf(system, custom_data)
vtf.writevcf(system, custom_data)
