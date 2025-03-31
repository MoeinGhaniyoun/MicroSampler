import matplotlib.pyplot as plt
import pickle
import sys
import re
import itertools
from matplotlib import rcParams
from Microarchitecture import *

component_names = {Component.LQ: 'Load Queue', Component.SQ: 'Store Queue', Component.ROB: 'Reorder Buffer', Component.LFB: 'Line Fill Buffer', \
                   Component.HWPREFETCHER: 'Next-Line Hardware Prefetcher', \
                   Component.EXESTATUS: 'Execution Unit Utilization'}
plot_blacklist = [] #Component.LFB,Component.HWPREFETCHER]
colors = ['blue', 'darkorange', 'violet', 'green', 'maroon', 'red', 'dimgray', 'plum', 'navy', 'steelblue', 'lightgreen', 'olivedrab', 'teal', 'lightpink', 'firebrick', 'brown']
markers = ['o', 'x', '<', '|', '*', '+', '_', '.', '^', 'P', 'X', 'D', '>', '1', '2', '3'] 

def uarch_diff(component, theta_lst, axs, _phi, _alpha):
    states = list()   
    seed = list(theta_lst.keys())[0]
    for state in theta_lst[seed][component]:
        states.append(state[0])

    for dclass in theta_lst.keys():
        if dclass == seed:
            continue
        for state in theta_lst[dclass][component]:
            sidx = find_index(states, lambda e: e.compare(component, state[0])) 
            if sidx is None:
                states.append(state[0])

    obsrv = {k: [] for k in theta_lst.keys()}
    diff = list()
    stats = [[] for _ in range(len(states))]
    for j in range(len(states)):
        for dclass in theta_lst:
            sidx = find_index(theta_lst[dclass][component], lambda e: e[0].compare(component, states[j]))
            if sidx is None:
                obsrv[dclass].append(0) 
                stats[j].append((dclass,0))
            else:
                obsrv[dclass].append(theta_lst[dclass][component][sidx][1])
                stats[j].append((dclass, theta_lst[dclass][component][sidx][1]))

    obsrv_candidates = {k: [] for k in theta_lst.keys()}
    for dclass in theta_lst:
        obsrv_candidates[dclass] = [0] * len(states) 

    for j in range(len(states)):
        candidate = None 
        alpha_violation = False
        for freq in stats[j]:
            if candidate and freq[1] >= _alpha:
                candidate = None
                break
            elif freq[1] >= _phi and not alpha_violation: 
                candidate = freq
            elif freq[1] >= _alpha:
                alpha_violation = True
        if candidate:
            class_lbl = candidate[0]
            candidate_freq = candidate[1]
            diff.append((class_lbl, states[j]))
            obsrv_candidates[class_lbl][j] = candidate_freq 


    if component not in plot_blacklist:
        axs.flat[component.value].axhspan(_phi, 1, color='green', alpha=0.15, label=r'$\phi$')
        axs.flat[component.value].axhspan(0, _alpha, color='red', alpha=0.15, label=r'$\alpha$')
        axs.flat[component.value].set_title(component_names[component])
        idx = 0
        for dclass in theta_lst.keys():
            #faceclr = 'none' if markers[idx] == 'o' else colors[idx]
            faceclr = colors[idx]
            axs.flat[component.value].scatter([j for j in range(len(states)) if obsrv[dclass][j] > 0.0], [obsrv[dclass][j] for j in range(len(states)) if obsrv[dclass][j] > 0.0], color=colors[idx], marker=markers[idx], facecolors=faceclr, label='e='+dclass)
            #axs.flat[component.value].scatter([j for j in range(len(states)) if obsrv_candidates[dclass][j] > 0.0], [obsrv_candidates[dclass][j] for j in range(len(states)) if obsrv_candidates[dclass][j] > 0.0], color='gold', marker='*', edgecolors='black', s=90)
            idx = idx + 1

    return diff

finUarch = open(sys.argv[1], 'rb')
finKey = open(sys.argv[2], 'r')
instructions, loops, states = pickle.load(finUarch)
_phi = float(sys.argv[5])
_alpha = float(sys.argv[6])
window = int(sys.argv[7])
iters = int(sys.argv[8])
n_classes = 2**window
print('phi: {}, alpha: {}'.format(_phi, _alpha))
print('window: {}'.format(window))

keyhex = finKey.read()
print(keyhex)
keystr = ''
for c in range(round(len(loops)*window/4)): #26 #100/8*2
    keystr += '{:04b}'.format(int(keyhex[c], 16))
keystr = keystr[::-1]
print(keystr)

key = []
for x in range(0, len(keystr), window):
    key.append(keystr[x:x+window])
print(key)

classcnt = {w: key.count(w) for w in key} 
print(classcnt)

loopsUArch = []
loop_unique_states = []
theta_lst = {w: {} for w in key}
diff = {}

# Cross-reference RoI from begin/end instructions with UArch objects
# for all cycles enclosed by the fetch/retire cycles for those RoI instructions.
print("Gathering loop state samples..")
for x in range(len(loops)):
    assert loops[x][-1].retire > 0

    fetch = loops[x][0].fetch
    retire = loops[x][-1].retire
    loopsUArch.append([state for state in states if state.cycle_begin >= fetch and state.cycle_begin <= retire])

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15,6))

assert len(loops) == len(loopsUArch)

print('iters: {}, loops: {}'.format(iters, len(loopsUArch)))

for component in Component:
    #  Collect common elements across rounds with the same key bit value
    for dclass in theta_lst:
        theta_lst[dclass][component] = list()
    #print("Finding unique and common elements for "+str(component))
    for idx in range(int(iters/window)):
        #if component == Component.EXESTATUS:
        #    print('ITER {}'.format(idx))
        # For each UArch state object associated with the current loop iteration...
        for state in loopsUArch[idx]:
            # If this state object differs from any of the state objects we have seen so far,
            # in terms of the given 'component', add it to the list of unqiue state objects encountered for that component.
            # This is accomplished by comparing the equivalence criterion for 'component' with all of the 
            # previously collected state objects for that component. 'sidx' indicates if a match was found for an "equivalent" state,
            # and a tally is updated to reflect how many repititions of the given (component,state) have been seen for this iteration.
            sidx = find_index(loop_unique_states, lambda e: e[0].compare(component, state))
            #if component == Component.EXESTATUS:
            #    print(state.print_feature(Component.EXESTATUS), sidx)
            if sidx is None:
                loop_unique_states.append([state, 1])
            else:
                loop_unique_states[sidx][1] = loop_unique_states[sidx][1] + 1

        # Now, update global tallies in states_bit1/0 for all unique states, with respect to 'component',
        # across all iterations. The tally per state must be <= 100, as there are 100 rounds being analyzed,
        # and is performed seperately for rounds with k=0 and k=1.
        # State tallies > 1 indicate that a given state is found in more than one iteration.
        # When comparing state across rounds with k=1/0, we may want to consider all states where
        # the tally count among iterations is beyond a certain threshold, say 75%.
        # We can say that, a state with a global tally >= 0.90 is representative of all iterations with k=1/0.
        for state in loop_unique_states:
            sidx = find_index(theta_lst[key[idx]][component], lambda e: e[0].compare(component, state[0]))
            if sidx is None:
                theta_lst[key[idx]][component].append([state[0], 1])
            else:
                theta_lst[key[idx]][component][sidx][1] = theta_lst[key[idx]][component][sidx][1] + 1

        del loop_unique_states[:]

    #Normalize tally by total class count
    for dclass in theta_lst:
        for state in theta_lst[dclass][component]:
            state[1] = state[1]/classcnt[dclass]

    diff[component] = uarch_diff(component, theta_lst, axs, _phi, _alpha)
                
    #for dclass in theta_lst.keys():
    #    print('Unique states for dclass: '+dclass)
    #    print(len(theta_lst[dclass][component]))
    #    for state in theta_lst[dclass][component]:
    #        print(state[0], state[1])

    print("=======================================================================================")
    print("=========== Diff of "+str(component_names[component])+" States (Candidates)  ==========")
    print("=======================================================================================")
    
    print('len: '+str(len(diff[component])))
    if len(diff[component]) == 0:
        print('No candidates found for '+str(component)+'!')
    for candidate in diff[component]:
        print('dclass: '+candidate[0])
        print('----------')
        print(candidate[1].print_feature(component))
        print('----------')

for i, row in enumerate(axs):
    for j, cell in enumerate(row):
        if i == len(axs) - 1:
            cell.set_xlabel("States")
        if j == 0:
            cell.set_ylabel("Iteration Frequency")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
handles, labels = axs.flat[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', fancybox=True, shadow=True, ncol=4, mode='expand')
plt.savefig(sys.argv[4]+'/stats-'+str(_phi)+'_'+str(_alpha)+'.pdf', bbox_inches='tight')

dtlbm = [0] * int(iters/window) 
dcachem = [0] * int(iters/window) 
fmiss = open(sys.argv[4]+'/miss_stats.csv', 'w+')
fmiss.write('ITERN,STATEN,MISSN,TYPE,PC,PADDR\n')
for idx in range(int(iters/window)):
    for s in range(len(loopsUArch[idx])):
        dtlbm[idx] += loopsUArch[idx][s].dtlbMisses.num_misses()
        dcachem[idx] += loopsUArch[idx][s].dcacheMisses.num_misses() 
        for j in range(loopsUArch[idx][s].dtlbMisses.num_misses()):
            fmiss.write(str(idx)+','+str(s)+','+str(j)+','+'DTLB,'+loopsUArch[idx][s].dtlbMisses.pc[j]+','+loopsUArch[idx][s].dtlbMisses.paddr[j]+'\n')
        for j in range(loopsUArch[idx][s].dcacheMisses.num_misses()):
            fmiss.write(str(idx)+','+str(s)+','+str(j)+','+'DCACHE,'+loopsUArch[idx][s].dcacheMisses.pc[j]+'\n')
fmiss.close()

print('+++++++++++++++++++++++++++++++')
for idx in range(int(iters/window)):
    for s in range(len(loopsUArch[idx])):
        print('loop: {}, state: {}, exe_unit {}'.format(idx, s, loopsUArch[idx][s].executionUnits.exeReqs))

print('+++++++++++++++++++++++++++++++')

print('Size of the loop state lists:')
print([len(loop) for loop in loopsUArch])
print('Counters for no. loops with each bit value:')
for dclass in theta_lst:
    print(classcnt[dclass])
print('no. cycles for each loop:')
print([loop[-1].retire-loop[0].fetch for loop in loops])
for x in range(len(loops)):
    print('iter: {}, cycle_begin: {}, cycle_end: {}'.format(x, loops[x][0].fetch, loops[x][-1].retire))

print('dtlb misses for each loop:')
print([dtlbm[x] for x in range(int(iters/window))])
print('dcache misses for each loop:')
print([dcachem[x] for x in range(int(iters/window))])

#for dclass in theta_lst.keys():
#    print(dclass)
#    for state in theta_lst[dclass][Component.EXESTATUS]:
#        print(state[0].print_feature(Component.EXESTATUS))
#        print(state[1])

#print(sum([len(loopsUArch[idx]) for idx in range(len(loops)-1) if key[idx] == '1'])/loop_bit1_cnt)
#print(sum([len(loopsUArch[idx]) for idx in range(len(loops)-1) if key[idx] == '0'])/loop_bit0_cnt)

for component in Component:
    print(str(component))
    for dclass in theta_lst:
        print('\t0x'+dclass+':'+ \
        str(len(theta_lst[dclass][component]))+','+ \
        str(len([x for x in theta_lst[dclass][component] if x[1] >= _phi]))+','+ \
        str(len([x for x in theta_lst[dclass][component] if x[1] <= _alpha]))+','+ \
        str(len([x for x in diff[component] if x[0] == dclass])))
        
with open(sys.argv[3], 'wb') as f:
    pickle.dump((loopsUArch, theta_lst, diff, key), f)

print('done.')
