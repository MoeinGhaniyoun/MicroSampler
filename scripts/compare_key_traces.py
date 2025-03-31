import os
import pickle
import sys
import re
import os
from datetime import datetime
from Microarchitecture import *

design = sys.argv[1]
suite = sys.argv[2]
app = sys.argv[3]
iters = sys.argv[4] 

now = datetime.now()
datestr = now.strftime("%m%d%Y")
filename = 'logs/'+design+'/compare/'+datestr+'/'+suite+'_'+app+'.log'
os.makedirs(os.path.dirname(filename), exist_ok=True)
fout = open(filename, 'w')
diff_regex = re.compile('logs/'+design+'/'+suite+'/'+app+'/'+iters+'/([\.\_\-x0-9a-z]+)/sets.pickle')
uarch_regex = re.compile('logs/'+design+'/'+suite+'/'+app+'/'+iters+'/([\.\_\-x0-9a-z]+)/uarch.pickle')

blob = os.popen('ls -llarth logs/'+design+'/'+suite+'/'+app+'/'+iters+'/*/sets.pickle')
lines = blob.readlines()
diffs = {}
loop_states = {}
states = {}

for line in lines:
    fname = line.split(' ')[-1].strip()
    match = re.search(diff_regex, fname)
    keyname = match.group(1)
    fout.write('Loading from '+fname+'...\n')
    loopsUArch, theta_lst, diff, keyval = pickle.load(open(fname, 'rb'))
    loop_states[keyname] = loopsUArch
    diffs[keyname] = diff

for component in Component:
    fout.write(str(component)+'\n')
    for key in diffs:
        fout.write('Key: '+key+'\n')
        fout.write(str(len(diffs[key][component]))+'\n')
        for state in diffs[key][component]:
            if component == Component.LQ:
                fout.write(state[1].lq.__str__()+'\n')
            elif component == Component.SQ:
                fout.write(state[1].sq.__str__()+'\n')
            elif component == Component.ROB:
                fout.write(state[1].rob.__str__()+'\n')
            elif component == Component.LFB:
                fout.write(state[1].lfb.__str__()+'\n')
            elif component == Component.HWPREFETCHER:
                fout.write(state[1].hwprefetcher.__str__()+'\n')
            elif component == Component.EXESTATUS:
                fout.write(state[1].executionUnits.__str__()+'\n')

global_diff = {comp: [] for comp in Component}
for component in Component:
    for key in diffs:
        for state in diffs[key][component]:
            idx = find_index(global_diff[component], lambda e: e[0].compare(component, state[1]))
            if idx is None:
                global_diff[component].append([state[1], 1])
            else:
                global_diff[component][idx][1] = global_diff[component][idx][1] + 1

for comp in Component:
    fout.write(str(comp)+'\n')
    tot = len(global_diff[comp])
    fout.write('nKeys: '+str(len(diffs))+'\n')
    if tot == 0:
        continue
    inv = len([x for x in global_diff[comp] if x[1] == len(diffs)])
    fout.write(str(tot) + ' ' + str(inv) + ' ' + str(float(inv/tot))+'\n')
    for state in global_diff[comp]:
        if comp == Component.LQ:
            fout.write(str(state[1])+': '), fout.write(state[0].lq.__str__()+'\n')
        elif comp == Component.SQ:
            fout.write(str(state[1])+': '), fout.write(state[0].sq.__str__()+'\n')
        elif comp == Component.ROB:
            fout.write(str(state[1])+': '), fout.write(state[0].rob.__str__()+'\n')
        elif comp == Component.LFB:
            fout.write(str(state[1])+': '), fout.write(state[0].lfb.__str__()+'\n')
        elif comp == Component.HWPREFETCHER:
            fout.write(str(state[1])+': '), fout.write(state[0].hwprefetcher.__str__()+'\n') 
        elif comp ==  Component.EXESTATUS:
            fout.write(str(state[1])+': '), fout.write(state[0].executionUnits.__str__()+'\n')

