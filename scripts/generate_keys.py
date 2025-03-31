import numpy
import sys

def write_weighted_key_bits(population, num):
    for weight in numpy.arange(0.1, 1, 0.1):
        for version in range(num):
            fout = open('keys/rand-'+"{:.2f}".format(round(weight, 1))+'_'+"{:.2f}".format(round(1-weight, 1))+'.v'+str(version)+'.key', 'w')
            samples = numpy.random.choice(population, size=1024, p=[weight, 1-weight])
            fout.write(hex(int(''.join(samples), 2))[2:])


def write_key_windows(groups, groupsize, keysize):
    keyval = ''
    #for g in range(groups):
    #    samples = numpy.random.choice(population, size=groupsize, p=[.5, .5])    
    #    grp_dict[g] = hex(int(''.join(samples), 2))[2:]
    #    print(grp_dict[g])

    grp_choice = numpy.random.choice([i for i in range(len(groups))], size=int(keysize/groupsize), p=[float(1/len(groups))] * len(groups))
    for win in grp_choice:
        keyval += groups[win]

    rem = 1024-len(keyval)
    keyval += '0' * rem

    print(hex(int(keyval, 2))[2:])

def write_strided_bitpattern(bits, keysize):
    keyval = ''
    size = int(keysize/len(bits))
    for i in range(size):
        keyval += bits

    rem = keysize-len(keyval)
    keyval += '0' * rem

    print(hex(int(keyval, 2))[2:])

grp5 = {0: '10010', 1: '11001', 2: '01000', 3: '10000', 4: '11100', 5: '00011', 6: '10100', 7: '10111', 8: '00001', 9: '10101'}
grp4 = {0: '1001', 1: '1010', 2: '1111', 3: '0000', 4: '1100', 5: '0011', 6: '0101', 7: '1000', 8: '0100', 9: '0010'}
if sys.argv[1] == 'rand':
    write_weighted_key_bits(['0', '1'], 10)
elif sys.argv[1] == 'window':
    if sys.argv[2] == '4':
        write_key_windows(grp4, 4, 1024)
    elif sys.argv[2] == '5':
        write_key_windows(grp5, 5, 1024)
elif sys.argv[1] == 'pattern':
    write_strided_bitpattern(sys.argv[2], 1024)
