import pandas as pd
import numpy
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Set plot config - APS style
plt.rcParams['figure.dpi']     = 100
plt.rcParams['font.size']      = 17
plt.rcParams['axes.linewidth'] = 1.25

plt.rcParams['font.weight']        = 'normal'
plt.rcParams['axes.labelweight']   = 'normal'
plt.rcParams['axes.titleweight']   = 'normal'

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc']     = 'upper center'

plt.rcParams['xtick.labelsize']     = 15
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['xtick.major.width']   = 1.25
plt.rcParams['xtick.major.size']    = 5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.minor.width']   = 1.25
plt.rcParams['xtick.minor.size']    = 3.5
plt.rcParams['xtick.top']           = 'on'

plt.rcParams['ytick.labelsize']     = 15
plt.rcParams['ytick.direction']     = 'in'
plt.rcParams['ytick.major.width']   = 1.25
plt.rcParams['ytick.major.size']    = 5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.width']   = 1.25
plt.rcParams['ytick.minor.size']    = 3.5
plt.rcParams['ytick.right']         = 'on'

# Latex options
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

def basis_change(x, nx, ny, nz):
  """ Computes the new coordinates of a 3D vector in a new
  basis set composed by the orthonormal vector nx, ny, and nz. 
  Args: 
    x: particles positions (np, 3) 
    nx, ny, nz: orthonormal vectors with shape (3)
  Returns:
    rotated particle positions (np, 3) """
  x_new = jnp.dot(x, nx)
  y_new = jnp.dot(x, ny)
  z_new = jnp.dot(x, nz)
  return jnp.array([x_new, y_new, z_new]).T
vbasis_change = vmap(basis_change, in_axes=(0,None,None,None), out_axes=0)
vbasis_change.__doc__ = 'Vmaped version of basis change'

def fbody_basis(xref):
  id = numpy.argmax(numpy.linalg.norm(xref, axis=-1))
  x_max = xref[id,:]
  nz = x_max/numpy.linalg.norm(x_max)
  dr = numpy.linalg.norm(xref[id,:]-xref, axis=-1)
  id2 = numpy.argmin(dr[numpy.nonzero(dr)])
  j = id2 if id2<id else id2+1
  ny = numpy.cross(nz, xref[j,:])/numpy.linalg.norm(numpy.cross(nz, xref[j,:]))
  nx = -numpy.cross(nz, ny)
  return nx, ny, nz

def global_3d_rotation(data, np, alpha, beta, gamma):
  """
  Perform a global rotation in a set of 3D particles coordinates
  Args:
    data: positions of the particles in the form [x1, y1, z1, x2, y2, z2, ...]
    np: number of particles
    alpha, beta, gamma: euler angles for the rotation
  Returns:
    Rotated data
  """
  data = jnp.reshape(data, [np, 3])
  rot_alpha = jnp.array([[jnp.cos(alpha), -jnp.sin(alpha), 0.], [jnp.sin(alpha), jnp.cos(alpha), 0.], [0., 0., 1.]])
  rot_beta  = jnp.array([[ jnp.cos(beta), 0.,  jnp.sin(beta)], [0., 1., 0.], [ -jnp.sin(beta), 0.,  jnp.cos(beta)]])
  rot_gamma = jnp.array([[1., 0., 0.], [0., jnp.cos(gamma), -jnp.sin(gamma)], [0., jnp.sin(gamma), jnp.cos(gamma)]])
  rotation = jnp.matmul(rot_alpha, np.matmul(rot_beta, rot_gamma))
  return jnp.matmul(data, rotation)
vrotation = vmap(global_3d_rotation, in_axes=(0,None,None,None,None), out_axes=0)
vrotation.__doc__ = 'Vmaped version of global 3d rotation'

def spherical_angles(v):
   """
   Computes the spherical coordinates angles theta and phi for 
   a 3D vector.
   """
   x, y, z = v
   xy = jnp.sqrt(x**2 + y**2)
   r = jnp.sqrt(xy**2 + z**2)
   theta = jnp.arccos(z/r)
   phi = jnp.arctan2(y, x)
#   theta = jnp.arctan2(y, x)
#   phi = jnp.arctan2(xy, z)
   return theta, phi
v_sph_ang = vmap(spherical_angles)
v_sph_ang.__doc__ = 'Vmaped version of spherical angles'

def centering(data, np, ndim):
  """
  Centering of the coordinates relative to the center of mass of equimass particles.
  Args:
    data: positions of the particles in the form [x1, y1, z1, x2, y2, z2, ...]
    np: number of particles
    ndim: number of spacial dimensions
  Returns:
    q: centered coordinates
  """
  x = jnp.reshape(data, [np, ndim])
  q = x - x.mean(axis=0, keepdims=True)
  return q
vcentering = vmap(centering, in_axes=(0,None,None), out_axes=0)
vcentering.__doc__ = 'Vmaped version of centering'

def profile_distance(data, np, ndim):
  """
  Compute the relative coordinates.
  Args:
    data: positions of the particles in the form [x1, y1, z1, x2, y2, z2, ...]
    np: number of particles
    ndim: number of spacial dimensions
  Returns:
    r: distance of particles, where r[i] is the distance of the ith particle
  """
  q = jnp.reshape(data, [np, ndim])
  r = jnp.linalg.norm(q, axis=-1)
  return r
vprofile_distance = vmap(profile_distance, in_axes=(0,None,None), out_axes=0)
vprofile_distance.__doc__ = 'Vmaped version of profile distance'

def relative_coordinates(data, np, ndim):
  """
  Compute the relative coordinates.
  Args:
    data: positions of the particles in the form [x1, y1, z1, x2, y2, z2, ...]
    np: number of particles
    ndim: number of spacial dimensions
  Returns:
    dq: matrices with the relative coordinates, where dq[i, j, :] is a vector with
        the relative coordinates x_{ij}, y_{ij}, z_{ij}
  """
  dq = jnp.reshape(data, [1, np, ndim]) - jnp.reshape(data, [np, 1, ndim])
  return dq
vrelative_coordinates = vmap(relative_coordinates, in_axes=(0,None,None), out_axes=0)
vrelative_coordinates.__doc__ = 'Vmaped version of relative coordinates'

def relative_distance(data, np, ndim):
  """
  Computes the relative distance between particles.
  Args:
    data: positions of the particles in the form [x1, y1, z1, x2, y2, z2, ...]
    np: number of particles
    ndim: number of spacial dimensions
  Returns:
    dr: matrix with the relative distance between particles, where dr[i,j] is
        the distance r_{ij} = |\vec r_i - \vec r_j|
  """
  dq = relative_coordinates(data, np, ndim)
  dr = jnp.linalg.norm(dq, axis=-1)
  return dr
vrelative_distance = vmap(relative_distance, in_axes=(0,None,None), out_axes=0)
vrelative_distance.__doc__ = 'Vmaped version of relative distance'

def distance_distribution(data, np, ndim, bins=1250, distr_type='profile', norm_type=None, density=True):
  """
  Computes distance distribution of samples.
  Args:
    data: samples of configurations in the shape (nsamples, np*ndim)
    np: number of particles
    ndim: number of spacial dimensions
    nbins: number of bins to consider in the histogram 
    distr_type: type of distribution, either 'pair' or 'profile', where pair computes 
                the pair distribution and profile computes the density profile
    norm_type: type of normalization, either 'volume', 'radial', or None.
    density: either to normalize the integral to one (true) or to N (false), where N is the number of
             particles if distr_type is 'profile' or the number of pairs of particles if
             distr_type is 'pair'.
  Returns:
    r: centered bins positions
    y: histogram values
  """
  if distr_type == 'profile':
    distance = vprofile_distance(data, np, ndim)
  elif distr_type == 'pair':
    distance = vrelative_distance(data, np, ndim)
    i, j = jnp.triu_indices(np, k=1)
    distance = distance[:, i, j]
  else:
    raise SystemExit("distr_type value not valid")

  nsamples, N = distance.shape
  distance = jnp.reshape(distance, [-1,])
  hist, x = jnp.histogram(distance, bins=bins, range=(0.0, distance.max()))
  r = 0.5 * (x[:-1] + x[1:])
  dr = x[1:] - x[:-1]

  if norm_type == 'volume':
    norm = 1.0 / (nsamples * 4.0 * jnp.pi * r**2 * dr)
  elif norm_type == 'radial':
    norm = 1.0 / (nsamples * r**2 * dr)
  elif norm_type == None:
    norm = 1.0 / (nsamples * dr)
  else:
    raise SystemExit("norm_type value not valid")

  if density:
    norm = norm / N

  y = hist / norm

  return r, y

def angular_distribution(data, bins=100, density=True):
  """
  Computes the angular distribution from 2d histogram.
  Args:
    data: samples of particles configurations with the shape (nsamples, np*ndim)
    bins: number of bins to consider
    density: either to normalize the area to one or not
  Returns:
    hist: 2d histogram of the angle distribution
    xedges, yedges: edges of the histogram
  """
  theta, phi = v_sph_ang(data.reshape([-1,3]))
  hist, xedges, yedges = jnp.histogram2d(phi, theta, bins=bins, density=density)
  return hist, xedges, yedges

def estimation(ene):
  """
  Computes the estimation of an observable and its standard deviation
  for correlated samples (Monte Carlo way)
  Args:
    ene: vector with the block averages
  Returns:
    ave: average of the vector
    std: standard deviation
  """
  ave = jnp.mean(ene)
  if ene.size > 1:
    std = jnp.sqrt((jnp.mean(ene**2)-jnp.mean(ene)**2)/(ene.size-1))
  else:
    std = jnp.nan
  return ave, std

def blocking(data, block_sizes):
  """
  Perform bloking procedure for estimation of integrals in a
  Monte Carlo way.
  Args:
    data: vector with a sequence of values to be blocked
    block_sizes: list with the size of blocks to be considered
  Returns:
    ave_dt: average of data
    vsd: array with same size as block_sizes with the standard deviation
         according to the size of the block considered
  """
  vsd = jnp.zeros(0)
  nsteps = jnp.size(data)
  for j in block_sizes:
    nblocks = nsteps//j
    data_block = jnp.mean(data.reshape([nblocks, j]), axis=1)
    ave_dt, std_dt = estimation(data_block)
    vsd = jnp.append(vsd, std_dt)
  return ave_dt, vsd

def scatter_plot(x, y,
                 figsize=[5.5, 4.5],
                 xlabel='x',
                 ylabel='y',
                 xlim=None,
                 ylim=None,
                 color=colors[0],
                 marker='.',
                 size=5):
  fig, ax = plt.subplots(figsize=figsize, dpi=300)
  ax.scatter(x, y, marker=marker, c=color, s=size)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if xlim: ax.set_xlim(xlim)
  if ylim: ax.set_ylim(ylim)
  return fig, ax

def angular_distribution_plot(hist, xedges, yedges, figsize=[5.5,4.5], cmap='inferno', legend=''):
  fig, ax = plt.subplots(figsize=figsize, dpi=300)
  xticks = jnp.linspace(-jnp.pi, jnp.pi, 9)  # 9 ticks from -pi to pi
  xlabels = ['$-\pi$',
             '$-\\frac{3\pi}{4}$',
             '$-\\frac{\pi}{2}$',
             '$-\\frac{\pi}{4}$',
             '$0$',
             '$\\frac{\pi}{4}$',
             '$\\frac{\pi}{2}$',
             '$\\frac{3\pi}{4}$',
             '$\pi$']
  yticks = jnp.linspace(0, jnp.pi, 5)  # 9 ticks from -pi to pi
  ylabels = ['$0$',
           '$\\frac{\pi}{4}$',
           '$\\frac{\pi}{2}$',
           '$\\frac{3\pi}{4}$',
           '$\pi$']
  im = ax.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, aspect='auto')
  ax.set_xlabel('$\\phi$')
  ax.set_ylabel('$\\theta$')
  ax.set_xticks(xticks)
  ax.set_xticklabels(xlabels)
  ax.set_yticks(yticks)
  ax.set_yticklabels(ylabels)
  ax.minorticks_off()
  fig.colorbar(im, label=legend)
  return fig, ax

def read_data(filename, sep=',', skiprows=0, header='infer'):
  """
  Read data from file
  Args:
    filename: name of the file from which to read the data
    sep: separator between columns ('\s+' is useful for space separated data)
    skiprows: number of rows to skip in the beggining
    header: header for the columns data
  Returns:
    data: DataFrame with the read data
  """
  data = pd.read_csv(filename, sep=sep, skiprows=skiprows, header=header)
  return data

def smoothen_data(data, window=10, polyorder=3):
  """
  Smoothens data points
  Args:
    data: array with the data to be smoothened
    window: length of the filter window
    polyorder: order of the polynomial used to fit the samples
  Returns:
    smoothened data
  """
  smoothened = signal.sagvol_filter(data, window, polyorder)
  return smoothened
