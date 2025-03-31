import os
import pickle
import sys
import re
from Microarchitecture import *

fname = sys.argv[1]
table = open('table-'+sys.argv[2]+'.csv', 'w+')

traces = {}

i1, i2, s1, s2 = pickle.load(open(fname, 'rb'))
traces['baseline'] = s1
traces['opt'] = s2

header = 'LQ-OCPNCY,'
for x in range(8):
    header += 'LQ-PC-'+str(x)+','
for x in range(8):
    header += 'LQ-ADDR-'+str(x)+','
header += 'SQ-OCPNCY,'
for x in range(8):
    header += 'SQ-PC-'+str(x)+','
for x in range(8):
    header += 'SQ-ADDR-'+str(x)+','
header += 'ROB-OCPNCY,'
for x in range(32):
    header += 'ROB-PC-'+str(x)+','
for x in range(16):
    header += 'LFB-'+str(x)+','
for x in range(4):
    header += 'EUU-'+str(x)+','
header += 'PREF-ADDR,IMPL'

row = ''
table.write(header+'\n')
for trace_name in traces:
    for state in traces[trace_name]:
        row += str(state.lq.occupancy) + ','
        for x in range(8):
            try: row += str(hex(state.lq.pc[x])) + ','
            except: row += ','
        row = row[:-1] +','
        for x in range(8):
            try: row += str(hex(state.lq.address[x])) + ',' 
            except: row += ',' 
        row = row[:-1] +','
        row += str(state.sq.occupancy) + ','
        for x in range(8):
            try: row += str(hex(state.sq.pc[x])) + ',' 
            except: row += ',' 
        row = row[:-1] +','
        for x in range(8):
            try: row += str(hex(state.sq.address[x])) + ','
            except: row += ',' 
        row = row[:-1] +','
        row += str(state.rob.occupancy) + ','
        for x in range(32):
            try: row += str(hex(state.rob.pc[x])) + ',' 
            except: row += ',' 
        row = row[:-1] +','
        for x in range(16):
            try: row += str(hex(state.lfb.data[x])) + ',' 
            except: row += ',' 
        row = row[:-1] +','
        for x in range(4):
            try: row += str(state.executionUnitsBusy.reqs[x]) + ',' 
            except: row += ',' 
        row = row[:-1] +','
        row += hex(state.hwprefetcher.address) + ',' 
        row += trace_name 
        table.write(row+'\n')
        row = ''
