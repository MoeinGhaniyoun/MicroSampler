"""
  Parser.py
"""
import sys
import Instruction
from Microarchitecture import *
import re
import collections
import gzip

cycleRegex = re.compile('Cycle=\s*([0-9]+)')
loadQueueEntry = re.compile('LoadQueueEntry\[([0-9 ]+)\]: PC:([0-9a-fx]+) \(([BHT ])\) Address:([0-9a-fx]+) \(([V-])\), TLBMiss:([TF-]), Uncacheable:([TF-]+), Executed:([TF-]+), Ignored:([TF-]+), Succeeded:([TF-]+), OrderFail:([TF-]+), Observed:([TF-]+), STList:([0-9a-fx]+), STIdx:([0-9a-fx]+), STForwValid:([TF-]), STForwIdx:([0-9a-fx]+) \[SN:([ 0-9]+)\]')
storeQueueEntry = re.compile('StoreQueueEntry\[([0-9 ]+)\]: PC:([0-9a-fx]+) \(([BHT ])\)\(([C\s])\)\(([E\s])\) Address:([0-9a-fx]+) \(([V-])\), TLBMiss:[TF-], Data:([0-9a-fx]+), Committed:[TF-], Succeeded:[TF-] \[SN:([ 0-9]+)\]')
robEntry = re.compile('ROB\[([0-9 ]+)\]: ([BHT ]) (P| ) \(\((V|-)\)\((B|-)\)\((U|-)\) ([x0-9a-f]+) \[([a-z.0-9 ,()+-_]+)\] (E|-) \(d:[fCX-] p[0-9 ]+, bm:[0-9a-f ]+ sdt:[ 0-9]+\) \[SN:([ 0-9]+)\]')
robEntry2 = re.compile('ROB\[([0-9 ]+)\]: ([BHT ]) (P| ) \(\(([V-]+)\)\(([B-]+)\)\([U-]+\) ([x0-9a-f]+) ([x0-9a-f]+) (\[[a-z0-9\.\, \(\)\+\-\_]+\])(\[[a-z0-9\.\, \(\)\+\-\_]+\]) [E|-],[E|-] [ 0-9]+,[ 0-9]+ \(d:[fCX-] p[0-9 ]+, bm:[0-9a-f ]+ sdt:[ 0-9]+\) \[SN:([ 0-9]+)\]\(d:[fCX-] p[0-9 ]+, bm:[0-9a-f ]+ sdt:[ 0-9]+\) \[SN:([ 0-9]+)\]')
lineFillBufferEntry = re.compile('LineBufferEntry \[([ 0-9]+)\] = ([0-9a-f]+)')
hwprefetchEntry = re.compile('Prefetcher: Address:([0-9a-fx]+) Data:([0-9a-fx]+)')
timestamp = re.compile('([0-9]+); O3PipeView:([a-z]+):\s+([0-9]+)(:([0-9xa-f]+):[0-9]+:\s+([0-9]+):([a-z0-9\-\.\s\(\)\,\-\+]+))?')
fpRegisterFileEntry = re.compile('FPRF\[([ 0-9]+)\]: ([0-9a-f]+)')
intRegisterFileEntry = re.compile('IPRF\[([ 0-9]+)\]: ([0-9a-f]+)')
exeStatus = re.compile('([a-zA-Z]+)(Req|Res): PC:([0-9a-fx]+)')
#dtlbStatus = re.compile('IncommingTLBResp: PC:([0-9a-fx]+) DTLBMiss: \(([T-])\)  PhysicalAddress:([0-9a-fx]+) Miss:[T-] PageFaultExcep:[E-]\([LSI_]\) AccessExcep:[E-]\([LSI_]\) MissAlignedExcep:[E-]\([LSI_]\) Cacheable:[T-] Prefetchable:[T-]')
dtlbStatus = re.compile('DTLBMiss: PC:([0-9a-fx]+) PhysicalAddress:([0-9a-fx]+)')
dcacheStatus = re.compile('DCacheResp: PC:([0-9a-fx]+)  Data:([0-9a-fx]+)  \(([V-])\)  IsHella:  \([V-]\)  Miss:  \([V-]\)   Release:   \([V-]\)')


def get_iterations(instr_dict, lookbehind_pc, begin_pc, terminal_pcs):
    # 2D list of all instructions (retired and speculative) for each loop iteration
    # Def: RoI; region of interest
    iterations = []
    roi = False
    last_inst_pc = ""
    for seqnum, instr in instr_dict.items():
        if instr.pc == begin_pc and instr.retire > 0:
            if last_inst_pc == lookbehind_pc:
                roi = True
                iterations.append([])
        elif instr.pc in terminal_pcs and instr.retire > 0:
            if roi:
                iterations[-1].append(instr)
                roi = False

        if roi:
            iterations[-1].append(instr)

        last_inst_pc = instr.pc

    return iterations 

def get_trace_data(fin, begin_roi_pc, end_roi_pc):

    instructions = collections.OrderedDict() 
    states = [] 

    cycle = -1
    curCycle = None
    begin_roi = False
    roi_seqnum = -1

    for line in fin:

        line = line.decode('utf8')

        # Encountering an O3 timestamp
        if (match := re.search(timestamp, line)):
            seqnum = int(match.group(1))
            stage = match.group(2)
            cycle = int(match.group(3))
            pc = match.group(5)
            inst = match.group(7)     

            if stage == "fetch":
                if pc == begin_roi_pc:
                    if not begin_roi:
                        begin_roi = True
                        roi_seqnum = seqnum

            if begin_roi:
                try:
                    if stage == "fetch":
                        instructions[seqnum] = Instruction.Instr(seqnum, pc, inst, cycle)
                    elif stage == "decode":
                        instructions[seqnum].decode = cycle
                    elif stage == "rename":
                        instructions[seqnum].rename = cycle
                    elif stage == "dispatch":
                        instructions[seqnum].dispatch = cycle
                    elif stage == "issue":
                        instructions[seqnum].issue = cycle
                    elif stage == "complete":
                        instructions[seqnum].complete = cycle
                    elif stage == "retire":
                        instructions[seqnum].retire = cycle
                        if instructions[seqnum].pc == end_roi_pc:
                            break
                except KeyError:
                    pass

        if not begin_roi:
            continue

        # Encountered a new cycle, create UArch object and continue looping over lines in file
        # until finding another cycle marker
        elif (match := re.search(cycleRegex, line)):
            cycle = int(match.group(1))

            if curCycle:
                assert len(curCycle.lfb.data) == 16
                assert curCycle.rob.occupancy == len(curCycle.rob.pc)
                #if len(states) == 0 or not curCycle == states[-1]:
                curCycle.cycle_end = cycle
                states.append(curCycle)

            curCycle = UArch(cycle)
        
        elif curCycle is None:
            continue

        elif (match := re.search(loadQueueEntry, line)):
            idx = int(match.group(1))
            sn = int(match.group(17))
            pc = int(match.group(2), 16)
            ptr = match.group(3)
            address = int(match.group(4), 16)
            valid = match.group(5)

            curCycle.lq.sn.append(sn)
            curCycle.lq.ptr.append(ptr)
            curCycle.lq.pc.append(pc)
            curCycle.lq.address.append(address)
            curCycle.lq.valid.append(valid)

            if idx == 7:
                try:
                    curCycle.lq.setmetaData()
                except Exception as e:
                    print(str(e) + ' LQ; Cycle='+str(curCycle.cycle_begin), file=sys.stderr)
                    sys.exit()

        elif (match := re.search(storeQueueEntry, line)):
            idx = int(match.group(1))
            sn = int(match.group(9))
            pc = int(match.group(2), 16)
            ptr = match.group(3)
            address = int(match.group(6), 16)
            valid = match.group(7)
            data = int(match.group(8), 16)

            curCycle.sq.sn.append(sn)
            curCycle.sq.ptr.append(ptr)
            curCycle.sq.pc.append(pc)
            curCycle.sq.address.append(address)
            curCycle.sq.data.append(data)
            curCycle.sq.valid.append(valid)

            if idx == 7:
                try:
                    curCycle.sq.setmetaData() 
                except Exception as e:
                    print(str(e) + ' SQ; Cycle='+str(curCycle.cycle_begin), file=sys.stderr)
                    sys.exit()

        elif (match := re.search(robEntry, line)):
            idx = int(match.group(1))
            sn = int(match.group(10))
            ptr = match.group(2)
            pc = int(match.group(7), 16)
            inst = match.group(8)
            valid = match.group(4)

            curCycle.rob.sn.append(sn)
            curCycle.rob.ptr.append(ptr)
            curCycle.rob.pc.append(pc)
            curCycle.rob.inst.append(inst)
            curCycle.rob.valid.append(valid)

            if idx == 31:
                try:
                    curCycle.rob.setmetaData()
                except Exception as e:
                    print(str(e) + ' ROB; Cycle='+str(curCycle.cycle_begin), file=sys.stderr)
                    sys.exit()


        elif (match := re.search(robEntry2, line)):
            idx = int(match.group(1))
            ptr = match.group(2)
            pc1 = int(match.group(6), 16)
            pc2 = int(match.group(7), 16)
            inst1 = match.group(8)
            inst2 = match.group(9) 
            sn1 = int(match.group(10))
            sn2 = int(match.group(11))

            curCycle.rob.sn.append(sn1)
            curCycle.rob.sn.append(sn2)
            curCycle.rob.ptr.append(ptr)
            curCycle.rob.ptr.append(' ')
            curCycle.rob.pc.append(pc1)
            curCycle.rob.pc.append(pc2)
            curCycle.rob.inst.append(inst1)
            curCycle.rob.inst.append(inst2)

            if idx == 15:
                curCycle.rob.setmetaData()

        elif (match := re.search(lineFillBufferEntry, line)):
            idx = int(match.group(1))
            data = int(match.group(2), 16)

            curCycle.lfb.data.append(data)

        elif (match := re.search(hwprefetchEntry, line)):
            address = int(match.group(1), 16)
            data = int(match.group(2), 16)

            curCycle.hwprefetcher.address = address
            curCycle.hwprefetcher.data = data

        elif (match := re.search(intRegisterFileEntry, line)):
            idx = int(match.group(1))
            data = int(match.group(2), 16)

            curCycle.intRegFile.data.append(data)

        elif (match := re.search(fpRegisterFileEntry, line)):
            idx = int(match.group(1))
            data = int(match.group(2), 16)

            curCycle.fpRegFile.data.append(data)

        elif (match := re.search(exeStatus, line)):
            funcUnitType = match.group(1)
            statusType = match.group(2)
            pc = match.group(3) 
        
            if statusType == 'Req':
                if funcUnitType == 'ALU':
                   curCycle.executionUnits.exeReqs[FunctionalUnits.ALU].append(pc) 
                elif funcUnitType == 'AddrCalc':
                   curCycle.executionUnits.exeReqs[FunctionalUnits.ADDRGEN].append(pc)
                elif funcUnitType == 'Mul':
                   curCycle.executionUnits.exeReqs[FunctionalUnits.MUL].append(pc)
                elif funcUnitType == 'Div':
                   curCycle.executionUnits.exeReqs[FunctionalUnits.DIV].append(pc)

        elif (match := re.search(dtlbStatus, line)):
            pc = match.group(1)
            paddr = match.group(2)            
            if valid == 'T':
                curCycle.dtlbMisses.pc.append(pc)
                curCycle.dtlbMisses.paddr.append(paddr)

        elif (match := re.search(dcacheStatus, line)):
            pc = match.group(1)
            data = match.group(2)
            valid = match.group(3)
            if valid == 'V':
                curCycle.dcacheMisses.pc.append(pc)

    return instructions, states


