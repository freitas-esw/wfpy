import jax.numpy as jnp

# https://doi.org/10.1351/goldbook.E02032
e = 1.602176634e-19       # elementary charge [C]

# https://doi.org/10.1351/goldbook.P04685
h = 6.62607015e-34        # Planck constant [J s]
hbar = 0.5*h/jnp.pi       # redeuced Planck constant [J s] 

# https://doi.org/10.1351/goldbook.A00504
a0 = 5.29177249e-11       # Bohr radius [m]
Eh = 4.3597482e-18        # Hartree [J]

# https://doi.org/10.1351/goldbook.E02008
me = 9.1093897e-31        # electron rest mass [kg]

# https://doi.org/10.1351/goldbook.A00497
mu = 1.6605402e-27        # atomic mass constant [kg]

# https://doi.org/10.1351/goldbook.B00695
kb = 1.380649e-23         # Boltzmann constant [J / K]

# https://doi.org/10.1351/goldbook.P04508
eps = 8.854187817e-12     # permeability of vacuum [F / m]
kc = 1.0/(4.0*jnp.pi*eps) # Coulomb constant [N m2 / C2]

# https://doi.org/10.1515/pac-2019-0603
mHe = 4.002602            # Helium mass in atomic mass constant
