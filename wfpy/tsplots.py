import wfpy
import os
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt #remove that

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
			      xlabel='Energy [K]',
                              ylabel='Optimization steps',
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
			      xlabel='Potential energy [K]',
                              ylabel='Optimization steps',
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
                                    norm_type=None,       # Normalize integral of P(r)
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

def gbnnhc_snowball_plots(np, ndim):
  data = pd.read_csv('positions-n'+str(np)+'.csv', header=None).values
