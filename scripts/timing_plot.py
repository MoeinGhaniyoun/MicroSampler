import matplotlib.pyplot as plt 
import pickle
import sys
import re
from matplotlib import rcParams
from Microarchitecture import *

ext = None 
keyname = sys.argv[1]
suite = sys.argv[2]
if len(sys.argv) > 3:
    ext = sys.argv[3]

app_names = ['vuln', 'dummy', 'consttime']
if ext:
    app_names = [x + '_' + ext for x in app_names]

titles = ['(a)', '(b)', '(c)']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,8))

loop_timing = {}

for x in range(len(app_names)):
    loop_timing[x] = {'0': list(), '1': list()}
    loopsUArch, theta_lst, diff, key = pickle.load(open('logs/baseline/'+suite+'/'+app_names[x]+'/100/'+keyname+'/sets.pickle', 'rb'))
    #print(''.join(key))
    for j in range(len(loopsUArch)):
        loop_timing[x][key[j]].append(loopsUArch[j][-1].cycle_begin - loopsUArch[j][0].cycle_begin)
    del loopsUArch 
    del theta_lst
    del diff
    del key

for x in range(len(app_names)):
    axs[x].hist([loop_timing[x]['1'], loop_timing[x]['0']], color=['blue', 'orange'], label=[r'$\theta_1$', r'$\theta_0$'], edgecolor='black')
    axs[x].set_xlabel('cycles', fontsize=14)
    axs[x].set_title(titles[x], fontsize=14)
axs[0].set_ylabel('No. Iterations', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
handles, titles = axs.flat[-1].get_legend_handles_labels()
fig.legend(handles, titles, loc='upper left', fancybox=True, shadow=True, ncol=3, mode='expand')

plt.savefig('logs/baseline/'+suite+'/timings/timings_'+keyname+('_' + ext if ext else '')+'.pdf', bbox_inches='tight')
