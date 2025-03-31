class Instr:
    def __init__(self, seqnum, pc, inst, fetch):
        self.seqnum = seqnum
        self.pc = pc
        self.inst = inst
        self.fetch = fetch
        self.decode = 0 
        self.rename = 0
        self.dispatch = 0 
        self.issue = 0 
        self.complete = 0 
        self.retire = 0 

        self.CGREEN = "\33[32m"
        self.CVIOLET = "\33[35m"
        self.CEND = "\033[0m"
 
    def __str__(self):
        return "seq: " + str(self.seqnum) + \
                "; pc: " + self.CGREEN+self.pc+self.CEND + \
                "; inst: " + self.CVIOLET+self.inst+self.CEND + \
                "; fetch: " + str(self.fetch) + \
                "; decode: " + str(self.decode) + \
                "; rename: " + str(self.rename) + \
                "; dispatch: " + str(self.dispatch) + \
                "; issue: " + str(self.issue) + \
                "; complete: " + str(self.complete) + \
                "; retire: " + str(self.retire)
