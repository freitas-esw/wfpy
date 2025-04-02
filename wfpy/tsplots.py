import wfpy
import os
import pandas as pd
import jax.numpy as jnp

def bnnhc_opt_plots(smin=100, smax=500, scale='log', aspect=1):
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

  return 
 
def bnnhc_vmc_plots(smin=100, smax=500, scale='log', aspect=1):
  data = pd.read_csv('vmc_stats.csv')

  x = data['step'].values
  y_ene = data['energy'].values
  y_pot = data['potential'].values
  y_mov = data['pmove'].values
