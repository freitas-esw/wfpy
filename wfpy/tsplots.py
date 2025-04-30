import wfpy
import os
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt #remove that

from scipy import signal

def bnnhc_opt_plots(smin=100, smax=500, scale='log', aspect=0.9):
  fns = [fn for fn in os.listdir() if fn.startswith('train_stats')]
  data = pd.DataFrame()
  for fn in fns:
    data = pd.concat([data, pd.read_csv(fn)])
  data = data.sort_values(by='step', ascending=True)

  x = data['step'].values
  y_ene = data['energy'].values
  y_pot = data['potential'].values
  y_mov = data['pmove'].values

  i = int(x[-1]/10)+1
  ene = jnp.mean(y_ene[-i:])
  std = jnp.sqrt(jnp.var(y_ene[-i:])) 

  fig, ax = wfpy.scatter_plot(x, 
 			      y_ene, 
                              xlabel='Optimization steps',
			      ylabel='Energy [K]',
          		      ylim=[ene-smin*std,ene+smax*std])
  ax.set_xscale(scale)

  ax.set_box_aspect(aspect)
  fig.tight_layout()
  fig.savefig('opt_ene.png', transparent=True)
  plt.close(fig)

  pot = jnp.mean(y_pot[-i:])
  std = jnp.sqrt(jnp.var(y_pot[-i:])) 

  fig, ax = wfpy.scatter_plot(x, 
 			      y_pot, 
                              xlabel='Optimization steps',
			      ylabel='Potential energy [K]',
          		      ylim=[pot-smin*std,pot+smax*std])
  ax.set_xscale(scale)
  
  axtw = ax.twinx()
  axtw.scatter(x, y_mov, c='r', s=5)
  axtw.set_ylabel('Acceptation rate')
  
  ax.set_box_aspect(aspect)
  fig.tight_layout()
  fig.savefig('opt_pot.png', transparent=True)
  plt.close(fig)

  return 
 
def bnnhc_vmc_plots(smin=100, smax=500, aspect=0.9):
  data = pd.read_csv('vmc_stats.csv')

  x = data['step'].values
  y_ene = data['energy'].values
  y_pot = data['potential'].values
  y_mov = data['pmove'].values
  

  i = int(x[-1]/4)+1
  bs = wfpy.factorize(x[i:].size)[:-1]
  ene, std = wfpy.blocking(y_ene[i:], bs)

  ene_str = '$E_{'+str(x[-1]+1-i)+'} = '+wfpy.result_str(ene, max(std))+'$ [K]'

  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  ax.hist(y_ene, bins=int(jnp.sqrt(y_ene.size)))
  ax.set_xlabel('Energy')
  ax.set_ylabel('Counts')
  ax.set_box_aspect(aspect)
  fig.tight_layout()
  fig.savefig('energy_histogram.png', transparent=True)
  plt.close(fig)

  fig, ax = wfpy.scatter_plot(bs, std, xlabel='Block size', ylabel='Standard deviation [K]')
  xl = ax.get_xlim()
  yl = ax.get_ylim()
  ax.text(xl[0]+0.1*(xl[1]-xl[0]), yl[0]+0.1*(yl[1]-yl[0]), ene_str)
  ax.set_box_aspect(aspect)
  fig.tight_layout()
  fig.savefig('blocking_analysis.png', transparent=True)
  plt.close(fig)
  
  y = data['energy'].expanding().mean()
  fig, ax = wfpy.scatter_plot(x, y, xlabel='Monte Carlo steps', ylabel='Accumulated Energy [K]')
  ax.set_box_aspect(aspect)
  fig.tight_layout()
  fig.savefig('expanding_ene_average.png', transparent=True)
  plt.close(fig)

  return 

def gbnnhc_distribution_plots(np, ndim):
  data = pd.read_csv('positions-n'+str(np)+'.csv', header=None).values

  r, y = wfpy.distance_distribution(data, np, ndim, 
                                    bins=1250, 
                                    distr_type='profile', # Density profile distribution
                                    norm_type='dist',     # Normalize integral of P(r)
                                    density=False)        # Normalize to N
  integral = jnp.sum(jnp.mean(r[1:]-r[:-1])*y)
  if (abs(integral-np)>0.001): print('Not expected value for integral:', integral)

  fig, ax = wfpy.scatter_plot(r, y, xlabel='$r$ [$a_0$]', ylabel='$n(r)$ [$a_0^{-1}$]')
  fig.tight_layout()
  fig.savefig('density-profile.png', transparent=True)
  plt.close(fig) 

  r, y = wfpy.distance_distribution(data, np, ndim, 
                                    bins=1250, 
                                    distr_type='pair',    # Pair correlation distribution
                                    norm_type='radial',   # Normalize integral of r^2 P(r)
                                    density=True)         # Normalize to 1
  integral = jnp.sum(jnp.mean(r[1:]-r[:-1])*y*r**2)
  if (abs(integral-1.0)>0.001): print('Not expected value for integral:', integral)

  fig, ax = wfpy.scatter_plot(r, y, xlabel='$r$ [$a_0$]', ylabel='$\\rho(r)$ [$a_0^{-3}$]')
  fig.tight_layout()
  fig.savefig('pair-density-function.png', transparent=True)
  plt.close(fig)

  hist, xed, yed = wfpy.angular_distribution(data, bins=100, density=True)
  fig, ax = wfpy.angular_distribution_plot(hist, xed, yed, legend='Angular density') 
  fig.tight_layout()
  fig.savefig('angular-density.png', transparent=True)
  plt.close(fig)

  return 

def gbnnhc_snowball_plots(np, ndim, nwalkers):

  data = pd.read_csv('positions-n'+str(np)+'.csv', header=None).values
  print('Snowball structure for two shells')
  print('#walkers:', nwalkers, '; #dimensions:', ndim)
  print('data shape:', data.shape, '; #samples:', data.shape[0]/nwalkers, '\n')
 
  r, y = wfpy.distance_distribution(data, np, ndim, 
                                    bins=1250, 
                                    distr_type='profile', # Density profile distribution
                                    norm_type='dist',     # Normalize integral of P(r)
                                    density=False)        # Normalize to N

  integral = jnp.sum(jnp.mean(r[1:]-r[:-1])*y)
  print('Total number of particles:', np)
  print('Integral of the density profile:', integral, '\n')

  rsmooth = signal.savgol_filter(r, int(r.size/10), 3)
  ysmooth = signal.savgol_filter(y, int(y.size/10), 3)

  xpeaks = signal.find_peaks(ysmooth, distance=20, height=1.0)
  xshell = signal.find_peaks(-ysmooth, distance=20, height=(-1.0,-0.005))

  print('Density profile peaks:', r[xpeaks[0]])
  print('Density profile bottoms:', r[xshell[0]], '\n')

  fig, ax = wfpy.scatter_plot(r, y, xlabel='$r_{\\text{HeI}}$ [$a_0$]', ylabel='$n(r_{\\text{HeI}})$ [$a_0^{-1}$]')
  ax.scatter(r[xpeaks[0]], y[xpeaks[0]], s=155, marker='|', c=wfpy.colors[1])
  ax.scatter(r[xshell[0]], y[xshell[0]], s=155, marker='|', c=wfpy.colors[3])
  ax.set_xlim([0.0-0.15,15.0+0.15])
  ax.set_ylim([0.0-0.12,12.0+0.12])
  fig.tight_layout()
  fig.savefig('density-profile.png', transparent=True)
  plt.close(fig) 

  c1s = r < r[xshell[0]]
  c2s = r > r[xshell[0]]
  ave_n1s = jnp.sum(y[c1s]) * jnp.mean(r[c1s][1:] - r[c1s][:-1])
  ave_n2s = jnp.sum(y[c2s]) * jnp.mean(r[c2s][1:] - r[c2s][:-1])

  print('Integral of the density profile for 1st shell', ave_n1s)
  print('1st shell number of particles:', round(ave_n1s))
  print('Integral of the density profile for 2st shell', ave_n2s)
  print('2nd shell number of particles:', round(ave_n2s), '\n')

  # Correct orientation
  x = data.reshape([-1, nwalkers, np, ndim]) 
 # x = data.reshape([-1, np, ndim]) 
  x_ave = jnp.mean(x, axis=0)
  x_ave_r = jnp.linalg.norm(x_ave, axis=-1)
  x_ave_1s = jnp.where(x_ave_r[...,None] < r[xshell[0]], x_ave, jnp.inf)
  nx, ny, nz = wfpy.vfbody_basis(x_ave_1s)
 # nx, ny, nz = wfpy.fbody_basis(x_ave_1s)
 
  for i in range(nwalkers):
    x[:,i,...] = wfpy.vbasis_change(x[:,i,...], nx[i,...], ny[i,...], nz[i,...])
 # x = wfpy.vbasis_change(x, nx, ny, nz)

  x = x.reshape([-1, np, ndim])
  qr = wfpy.vprofile_distance(x, np, ndim)

  # Shell pair density distributions
  x_1s = jnp.where(qr[...,None] < r[xshell[0]], x, jnp.nan)
  dr_1s = wfpy.vrelative_distance(x_1s, np, ndim)
  dr_1s = dr_1s[~jnp.isnan(dr_1s)]
  r_1s, y_1s = wfpy.distribution_histogram(dr_1s[...,None],
                                    bins=1250, 
                                    norm_type='dist',    # Normalize integral of \rho(r)
                                    density=True)        # Normalize to 1

  integral = jnp.sum(jnp.mean(r_1s[1:]-r_1s[:-1])*y_1s)
  print('Integral of the 1st shell pair density distribution:', integral)

  rs_1s = signal.savgol_filter(r_1s, int(r_1s.size/10), 3)
  ys_1s = signal.savgol_filter(y_1s, int(y_1s.size/10), 3)
  rp_1s = wfpy.find_peaks(ys_1s, npeaks=2, distance=50, height=0.15)
  print('1st shell pair density distribution peaks:', rs_1s[rp_1s[0]])
  
  fig, ax = wfpy.scatter_plot(r_1s, y_1s, xlabel='$r_{\\text{HeHe}}$ [$a_0$]', ylabel='$r_{\\text{HeHe}}^2 \ \\rho_{1s}(r_{\\text{HeHe}})$ [$a_0^{-1}$]')
  ax.scatter(r_1s[rp_1s[0]], y_1s[rp_1s[0]], s=155, marker='|', c=wfpy.colors[1])
  ax.set_xlim([0.0-0.12,12.0+0.12])
  ax.set_ylim([0.0-0.00325, 0.325+0.00325])
  fig.tight_layout()
  fig.savefig('pair-density-1st-shell.png', transparent=True)
  plt.close(fig) 

  x_2s = jnp.where(qr[...,None] > r[xshell[0]], x, jnp.nan)
  dr_2s = wfpy.vrelative_distance(x_2s, np, ndim)
  dr_2s = dr_2s[~jnp.isnan(dr_2s)]
  r_2s, y_2s = wfpy.distribution_histogram(dr_2s[...,None],
                                    bins=1250, 
                                    norm_type='dist',    # Normalize integral of \rho(r)
                                    density=True)        # Normalize to 1

  integral = jnp.sum(jnp.mean(r_2s[1:]-r_2s[:-1])*y_2s)
  print('Integral of the 2nd shell pair density distribution:', integral)

  rs_2s = signal.savgol_filter(r_2s, int(r_2s.size/10), 3)
  ys_2s = signal.savgol_filter(y_2s, int(y_2s.size/10), 3)
  npeaks = 3 if round(ave_n1s) == 12 else 2 
  rp_2s = wfpy.find_peaks(ys_2s, npeaks=npeaks, distance=50, height=0.01)
  print('2st shell pair density distribution peaks:', rs_2s[rp_2s[0]], '\n')
  
  fig, ax = wfpy.scatter_plot(r_2s, y_2s, xlabel='$r_{\\text{HeHe}}$ [$a_0$]', ylabel='$r_{\\text{HeHe}}^2 \ \\rho_{2s}(r_{\\text{HeHe}})$ [$a_0^{-1}$]')
  ax.scatter(r_2s[rp_2s[0]], y_2s[rp_2s[0]], s=155, marker='|', c=wfpy.colors[1])
  ax.set_xlim([0.0-0.25,25.0+0.25])
  ax.set_ylim([0.0-0.0012,0.12+0.0012])
  fig.tight_layout()
  fig.savefig('pair-density-2st-shell.png', transparent=True)
  plt.close(fig) 

  # Angular histograms
 # plt.rcParams['font.size']       = 20
 # plt.rcParams['xtick.labelsize'] = 20
 # plt.rcParams['ytick.labelsize'] = 20

  q_1s = x_1s[~jnp.isnan(x_1s)].reshape([-1,3])
  h_1s, xed_1s, yed_1s = wfpy.angular_distribution(q_1s)
  fig, ax = wfpy.angular_distribution_plot(h_1s, xed_1s, yed_1s, legend='Angular density', vmin=0.0, vmax=1.2)
  fig.tight_layout()
  fig.savefig('angular-density-1st-shell.png', transparent=True)
  plt.close(fig) 

  q_2s = x_2s[~jnp.isnan(x_2s)].reshape([-1,3])
  h_2s, xed_2s, yed_2s = wfpy.angular_distribution(q_2s)
  fig, ax = wfpy.angular_distribution_plot(h_2s, xed_2s, yed_2s, legend='Angular density', vmin=0.0, vmax=0.34)
  fig.tight_layout()
  fig.savefig('angular-density-2st-shell.png', transparent=True)
  plt.close(fig) 

  return 

def distribution_plots(data, np, ndim,
                 figsize=[5.5, 4.5],
                 centering=False):
  """ """

  dq = wfpy.vrelative_coordinates(data, np, ndim)
  dr = jnp.linalg.norm(dq, axis=-1)
  i, j = jnp.triu_indices(np, k=1)

  dq = dq[:,i,j,:]
  dr = dr[:,i,j]

  dz, hdz = wfpy.distribution_histogram(dq[:,2], norm_type='dist', density=True, rmin=jnp.min(dq[:,2]))
  dy, hdy = wfpy.distribution_histogram(dq[:,1], norm_type='dist', density=True, rmin=jnp.min(dq[:,1]))
  dx, hdx = wfpy.distribution_histogram(dq[:,0], norm_type='dist', density=True, rmin=jnp.min(dq[:,0]))

  if centering:
    q = wfpy.vcentering(data, np, ndim)
  else:
    q = data.reshape([-1, np, ndim])

  print(dq.shape)
  print(q.shape)
  
  z, hz = wfpy.distribution_histogram(q[:,2], norm_type='dist', density=True, rmin=jnp.min(q[:,2]))
  y, hy = wfpy.distribution_histogram(q[:,1], norm_type='dist', density=True, rmin=jnp.min(q[:,1]))
  x, hx = wfpy.distribution_histogram(q[:,0], norm_type='dist', density=True, rmin=jnp.min(q[:,0]))

  fig, ax = wfpy.scatter_plot(dz, hdz, xlabel='$z_{ij}$', ylabel='$P(z_{ij})$ [$r_0^{-1}$]')
  fig.tight_layout()
  fig.savefig('pdz-distribution.svg', transparent=True)
  plt.close(fig)

  fig, ax = wfpy.scatter_plot(dy, hdy, xlabel='$y_{ij}$', ylabel='$P(y_{ij})$ [$r_0^{-1}$]')
  fig.tight_layout()
  fig.savefig('pdy-distribution.svg', transparent=True)
  plt.close(fig)

  fig, ax = wfpy.scatter_plot(dx, hdx, xlabel='$x_{ij}$', ylabel='$P(x_{ij})$ [$r_0^{-1}$]')
  fig.tight_layout()
  fig.savefig('pdx-distribution.svg', transparent=True)
  plt.close(fig)


  fig, ax = wfpy.scatter_plot(z, hz, xlabel='$z_{i}$', ylabel='$P(z_{i})$ [$r_0^{-1}$]')
  fig.tight_layout()
  fig.savefig('pz-distribution.svg', transparent=True)
  plt.close(fig)

  fig, ax = wfpy.scatter_plot(y, hy, xlabel='$y_{i}$', ylabel='$P(y_{i})$ [$r_0^{-1}$]')
  fig.tight_layout()
  fig.savefig('py-distribution.svg', transparent=True)
  plt.close(fig)

  fig, ax = wfpy.scatter_plot(x, hx, xlabel='$x_{i}$', ylabel='$P(x_{i})$ [$r_0^{-1}$]')
  fig.tight_layout()
  fig.savefig('px-distribution.svg', transparent=True)
  plt.close(fig)

 
  return
