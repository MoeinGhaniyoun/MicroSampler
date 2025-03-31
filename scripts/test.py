import pickle
import Instruction
from Microarchitecture import *

instructions, loops, states = pickle.load(open(sys.argv[1], 'rb'))


"""print(len(stateDeltas[Component.LFB]))

print(loops[0][0].fetch)
print(loops[0][-1].retire)
print(stateDeltas[Component.LFB][0].cycle_begin)
print(stateDeltas[Component.LFB][0].cycle_end)
print(stateDeltas[Component.LFB][1].cycle_begin)
print(stateDeltas[Component.LFB][1].cycle_end)

print([loop[-1].retire-loop[0].fetch for loop in loops])
"""

#print(stateDeltas[Component.LFB][100].lfb.data)
#testcase = UArch('fake')
#testcase.lfb.data = stateDeltas[Component.LFB][100].lfb.data[4:]+stateDeltas[Component.LFB][100].lfb.data[0:4]
#print(testcase.lfb.data)
#print(testcase.compare(Component.LFB, stateDeltas[Component.LFB][100]))
#testcase.lfb.data[1] = '0xdeadbeef'
#print(testcase.compare(Component.LFB, stateDeltas[Component.LFB][100]))
