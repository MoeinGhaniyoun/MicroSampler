import pickle
import Instruction
import Microarchitecture
import sys

fname = sys.argv[1]
instructions, loops, states = pickle.load(open(fname, 'rb')) 
