"""
 parse_trace.py: [trace_filename.gzip] [output_pickle_filename] [br_i31_modpow_begin] [main_end] [br_i31_modpow_call_ccopy_addr]
                 [ccopy_begin_addr] [ccopy_end_addr]
"""

import sys
import Instruction
import Microarchitecture
import re
import pickle
import collections
import gzip
from Parser import *

print('begin_roi: '+sys.argv[3]+', end_roi: '+sys.argv[4]+', lookbehind_pc: '+sys.argv[5]+', begin_pc: '+sys.argv[6]+', terminal_pcs: '+sys.argv[7])
finName = sys.argv[1]
fout = sys.argv[2]  
fin = gzip.open(finName, 'r')     
print('log: '+finName+'\n')
begin_roi = sys.argv[3]
end_roi = sys.argv[4]
lookbehind_pc = sys.argv[5]
begin_pc = sys.argv[6]
terminal_pcs = sys.argv[7]

instructions, states = get_trace_data(fin, begin_roi, end_roi)

loops = get_iterations(instructions, sys.argv[5], sys.argv[6], sys.argv[7])

with open(fout, 'wb') as f:
    pickle.dump((instructions, loops, states), f)

print('no. loops recorded: '+str(len(loops)))
print('Total no. uarch states collected: '+str(len(states)-1))
print('Total no. cycles in RoI: '+str(states[-1].cycle_begin - states[0].cycle_begin))

