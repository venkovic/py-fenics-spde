import numpy as np
import pylab as pl

pl.rcParams['text.usetex'] = True
params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}
pl.rcParams.update(params)

data_path = 'data/'
solve_path = 'data/solve/'
img_path = 'img/'

#
# Find rgb triplets for tab-colors at:
# https://public.tableau.com/views/TableauColors/ColorPaletteswithRGBValues?%3Aembed=y&%3AshowVizHome=no&%3Adisplay_count=y&%3Adisplay_static_image=y
#

def plot_results(theta, ratios, nmcmcs, nEl, smp_type='hybrid_md', with_current=False, fig_ext='png', dpi=300):
  """
  Plot PCG-AMG results for different symmetries.

  """

  #
  lw = 1.5
  #cols = ('tab:red', 'tab:blue', 'tab:green', 'tab:purple',)
  cols = ('grey', 'tab:purple', 'tab:green', 'tab:blue',)
  markers = ('s', 'o', '^', 'D',)
  fdict = {'fontsize':14}
  #
  #
  fig, axes = pl.subplots(1, 1, figsize=(4.3, 3.3))
  #
  iters_0 = np.zeros(len(nmcmcs))
  iters_t = np.zeros(len(nmcmcs))
  iters_d = np.zeros(len(nmcmcs))
  #
  for k, ratio in enumerate(ratios):
    #
    for i, nKL in enumerate(nmcmcs):
      #
      _smp_type = smp_type + str(nKL)
      iters_0[i] = np.load('%s%s_ratio%g_theta%g.%dDoFs.pcg-amg-median.iters.npy' % (solve_path, _smp_type, ratio, theta, nEl)).mean()
      iters_t[i] = np.load('%s%s_ratio%g_theta%g.%dDoFs.pcg-amg-current.iters.npy' % (solve_path, _smp_type, ratio, theta, nEl)).mean()
      iters_d[i] = np.load('%s%s_ratio%g_theta%g.%dDoFs.pcg-amg-deflated.iters.npy' % (solve_path, _smp_type, ratio, theta, nEl)).mean()
    #
    if with_current:
      axes.plot(nmcmcs, iters_0, '-' + markers[k], color=cols[k], lw=lw)
      axes.plot(nmcmcs, iters_t, '-.' + markers[k], color=cols[k], lw=lw)
      axes.plot(nmcmcs, iters_d, '--' + markers[k], color=cols[k], lw=lw)
    else:
      axes.plot(nmcmcs, iters_0, '-' + markers[0], color=cols[0], lw=lw, label='PCG (AMG)')
      axes.plot(nmcmcs, iters_d, '--' + markers[1], color=cols[1], lw=lw, label='Def-eigPCG (AMG)')



    #
    #axes.set_title(f'{nEl:,} DoFs')
  #
  #xlims = ax.get_xlim()
  #ax.set_xlim(xlims[1], xlims[0])
  axes.grid(linestyle="-.")
  axes.set_xlabel('KL modes sampled by MCMC', fontdict=fdict)
  #
  axes.set_ylabel(r'$\mathrm{Average}\; \#\; \mathrm{solver\ iterations}$', fontdict=fdict)
  #
  axes.tick_params(axis='both', which='major', labelsize=fdict['fontsize'])
  pl.legend()
  #
  #pl.subplots_adjust(wspace=.12)
  #
  pl.savefig('%sdef-pcg-amg.%s' % (img_path, fig_ext), bbox_inches='tight', dpi=dpi)
