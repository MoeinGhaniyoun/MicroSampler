# Microarchitectural Vulnerability Detection
This project investigates how best to utilize traces from pre-silicon simulation for vulnerability detection at the microarchitectural level of abstraction during the IC design life-cycle.  In particular, the analysis is focused on traces from executions of security critical applications in order to validate claims of properties enabling confidentiality protections.

The RISC-V BOOM processor released with the Chipyard framework from UC Berkeley is used as a testbed, along with cryptographic primitives from the BearSSL library as applications to study.

The tool has three stages: simulation, parsing and calculation of vulnerability metrics.

## Quick Start

**Dependencies**
1. Requires Python3 > 3.8 (suggest using <code>conda</code> to install local version)

Run <code>make</code> in <code>apps/bearssl-0.6/microsampler_tests</code> to compile all the tests.

The file <code>scripts/launch_runs.sh</code> is a job scheduling script for a local cluster. This can be used to launch multiple runs across nodes using SSH of the same application, selecting different inputs (keys) and hardware designs. This script is simply a helper-wrapper which then executes <code>do_simulation.sh</code>, <code>do_parse.sh</code> and <code>do_stats.sh</code> followed by <code>parse_trace.py</code> and <code>stats.py</code>, respectively.
The script should be called three times to launch the simulation, parsing and stats collection phases. Below are some examples of its use:  
> <code>./scripts/launch_runs.sh -action simulate -suite bearssl_synthetic -appsi v2 -design baseline -mode ssh</code>    

This will launch seperate simulations of the v2 application using each available key as input, defined in the <code>keys</code> array of <code>launch_runs.sh</code>

> <code>./scripts/launch_runs.sh -action simulate -suite bearssl_synthetic  **-appsi "v1 v2 v3"**  **-keysi 0xaa** -design baseline -mode ssh</code> 
   
This will launch a simulation only for the 0xaa input, for three applications (v1, v2 & v3)

> <code>./scripts/launch_runs.sh -action simulate -suite bearssl_synthetic -appsi v2 -design baseline **-mode dryrun**</code> 

Print the command that will be issued to the remote node over SSH, instead of running it

