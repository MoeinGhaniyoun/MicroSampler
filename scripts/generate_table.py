import os
import pickle
import sys
import re
from Microarchitecture import *

diff_regex = re.compile('logs/'+sys.argv[3]+'/'+sys.argv[2]+'/([0-9a-z_]+)/100/([\.\_\-x0-9a-z]+)/sets.pickle')

blob = os.popen('ls logs/'+sys.argv[3]+'/'+sys.argv[2]+'/'+sys.argv[1]+'/100/*/sets.pickle')
traces = blob.readlines()

if not os.path.exists('data/'+sys.argv[3]):
    os.makedirs('data/'+sys.argv[3])
table = open('data/'+sys.argv[3]+'/table-'+sys.argv[1]+'.csv', 'w')
iters = 100

diffs = {}
testnames = {}
seckeybin = {}

for trace in traces:
    fname = trace.strip()
    match = re.search(diff_regex, fname)
    keystr = match.group(2)
    print('Loading from '+fname+'...')
    loopsUArch, theta_lst, diff, keybin = pickle.load(open(fname, 'rb'))
    testnames[keystr] = loopsUArch
    seckeybin[keystr] = keybin
    diffs[keystr] = diff

header = 'CYCLE,'
header += 'LQ-OCPNCY,'
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
for x in FunctionalUnits:
    header += 'EUU-'+str(x.name)+'-0,'
    if x.name == 'ALU': 
        header += 'EUU-'+str(x.name)+'-1,'
header += 'PREF-ADDR,CLASS,ITER,KEY,DTLB-NMISS,DCACHE-NMISS'

row = ''
table.write(header+'\n')
for keystr in testnames:
    for loop in range(iters):
        for state in testnames[keystr][loop]:
            row += str(state.cycle_begin) + ','
            row += str(state.lq.occupancy) + ','
            for x in range(8):
                try: row += str(hex(state.lq.pc[x])) + ','
                except: row += ','
            row = row[:-1] + ','
            for x in range(8):
                try: row += str(hex(state.lq.address[x])) + ',' 
                except: row += ',' 
            row = row[:-1] + ','
            row += str(state.sq.occupancy) + ','
            for x in range(8):
                try: row += str(hex(state.sq.pc[x])) + ',' 
                except: row += ',' 
            row = row[:-1] + ','
            for x in range(8):
                try: row += str(hex(state.sq.address[x])) + ','
                except: row += ',' 
            row = row[:-1] + ','
            row += str(state.rob.occupancy) + ','
            for x in range(32):
                try: row += str(hex(state.rob.pc[x])) + ',' 
                except: row += ',' 
            row = row[:-1] + ','
            for x in range(16):
                try: row += str(hex(state.lfb.data[x])) + ',' 
                except: row += ',' 
            row = row[:-1] + ','
            for x in FunctionalUnits:
                try: row += str(state.executionUnits.exeReqs[x][0]) + ',' #one instruction in any unit in a cycle for SmallBoomConfig
                except: row += ','
                if x.name == 'ALU':
                    try: row += str(state.executionUnits.exeReqs[x][1]) + ',' #two ALU units/instructions in a cycle for SmallBoomConfig with fast-bypass
                    except: row += ','
            row = row[:-1] + ','
            row += hex(state.hwprefetcher.address) + ',' 
            row += seckeybin[keystr][loop] + ','
            row += str(loop) + ','
            row += keystr + ','
            row += str(state.dtlbMisses.num_misses()) + ','
            row += str(state.dcacheMisses.num_misses())
            table.write(row+'\n')

            row = ''

