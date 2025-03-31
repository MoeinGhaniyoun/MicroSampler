import sys
import re

"""
arg1: objdump of application binary
arg2: func1, br_i31_modpow + [_v1, _v2] + [_fence]
arg3: func2, br_ccopy [v1, v2]
arg4: [warmup]? 

"""

dissasmRegex = re.compile('([0-9a-f]+):\s+([0-9a-f]+)\s+([a-z]+)\s+([a-z0-9\,]+)(\s+<([a-z0-9_]+)>)?')
jmpAddrRegex = re.compile('ra,([0-9a-f]+)')
funcDefRegex = re.compile('([0-9a-f]+)(\s+<([a-z0-9_]+)>)?:')

fin = open(sys.argv[1], 'r')

marker_reached = True
funcdef_reached = False

if len(sys.argv) > 4:
    if sys.argv[4] == 'warmup':
        marker_reached = False

pc_lst = []
for line in fin:
    if (match := re.search(dissasmRegex, line)):
        pc = match.group(1)
        encoding = match.group(2)
        opcode = match.group(3)
        operands = match.group(4)
        func = match.group(6)
        if encoding == "00008013":
            marker_reached = True
            continue
        if marker_reached:
            if opcode == "jal":
                if func == sys.argv[2]:
                    pc_lst.append(pc)
                    pc_lst.append(hex(int(pc, 16) + 4)[2:])
                elif funcdef_reached and func == sys.argv[3]:
                    pc_lst.append(pc)
                    addr = re.search(jmpAddrRegex, operands).group(1)
                    pc_lst.append(addr)
                    pc_lst.append(hex(int(pc, 16) + 4)[2:])
                    break

    elif (match := re.search(funcDefRegex, line)):
        fname = match.group(3)
        if fname == sys.argv[2]:
            funcdef_reached = True
    

print(' '.join(['0x00000' + s for s in pc_lst]))

