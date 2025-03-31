import matplotlib.pyplot as plt
import pickle
import sys
from Microarchitecture import *

appname = sys.argv[1]
keyname = sys.argv[2]
uarch = sys.argv[3]
suite = sys.argv[4]
iters = 100
window = 1

titles = ['Iteration Latency', 'DTLB', 'DCache']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,8))

loop_timing = {'0': list(), '1': list()}
loop_dtlb_misses = {'0': list(), '1': list()}
loop_dcache_misses = {'0': list(), '1': list()}
instructions, loops, states = pickle.load(open('logs/'+uarch+'/'+suite+'/'+appname+'/100/'+keyname+'/uarch.pickle', 'rb')) 
loopsUArch, theta_lst, diff, key = pickle.load(open('logs/'+uarch+'/'+suite+'/'+appname+'/100/'+keyname+'/sets.pickle', 'rb'))
for j in range(int(iters/window)):
   loop_timing[key[j]].append(loops[j][-1].retire - loops[j][0].fetch) 
   dtlbm = 0
   dcachem = 0
   for ustate in loopsUArch[j]:
       dtlbm += ustate.dtlbMisses.num_misses()
       dcachem += ustate.dcacheMisses.num_misses()    
   loop_dtlb_misses[key[j]].append(dtlbm)
   loop_dcache_misses[key[j]].append(dcachem)

axs[0].hist([loop_timing['1'], loop_timing['0']], color=['blue', 'orange'], label=[r'$\theta_1$', r'$\theta_0$'], edgecolor='black')
axs[0].set_xlabel('Cycles', fontsize=14)
axs[0].set_title(titles[0], fontsize=14)
axs[1].hist([loop_dtlb_misses['1'], loop_dtlb_misses['0']], color=['blue', 'orange'], label=[r'$\theta_1$', r'$\theta_0$'], edgecolor='black')
axs[1].set_xlabel('Misses', fontsize=14)
axs[1].set_title(titles[1], fontsize=14)
axs[2].hist([loop_dcache_misses['1'], loop_dcache_misses['0']], color=['blue', 'orange'], label=[r'$\theta_1$', r'$\theta_0$'], edgecolor='black')
axs[2].set_xlabel('Misses', fontsize=14)
axs[2].set_title(titles[2], fontsize=14)
axs[0].set_ylabel('No. Iterations', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
handles, titles = axs.flat[-1].get_legend_handles_labels()
fig.legend(handles, titles, loc='upper left', fancybox=True, shadow=True, ncol=3, mode='expand')
plt.savefig('logs/'+uarch+'/'+suite+'/'+appname+'/100/'+keyname+'/timing_stats.pdf', bbox_inches='tight')
