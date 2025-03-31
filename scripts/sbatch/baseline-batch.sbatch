#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --job-name=microsampler
#SBATCH --account=PAS2222
#SBATCH --mem=4gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=barber.m.kristin@gmail.com

source ~/anaconda3/etc/profile.d/conda.sh
conda activate microsampler

#./scripts/launch_runs.sh -action simulate -suite bearssl_synthetic -appsi v1 -mode dryrun


### v1, 0xaa ###
mkdir -p logs/baseline/bearssl_synthetic/v1/100/0xaa; ./scripts/do_simulation.sh 0xaa bearssl_synthetic v1 100 baseline > logs/baseline/bearssl_synthetic/v1/100/0xaa/launch_simulation.log

./scripts/do_parse.sh 0xaa 0x00000105c8 0x00000105cc 0x0000010b72 0x00000106cc 0x0000010b76 bearssl_synthetic v1 100 baseline > logs/baseline/bearssl_synthetic/v1/100/0xaa/launch_parse.log

./scripts/do_stats.sh 0xaa bearssl_synthetic v1 .90 .10 1 100 baseline > logs/baseline/bearssl_synthetic/v1/100/0xaa/launch_stats.log

### v2, 0xaa ###
mkdir -p logs/baseline/bearssl_synthetic/v2/100/0xaa; ./scripts/do_simulation.sh 0xaa bearssl_synthetic v2 100 baseline > logs/baseline/bearssl_synthetic/v2/100/0xaa/launch_simulation.log

./scripts/do_parse.sh 0xaa 0x00000105cc 0x00000105d0 0x0000010d22 0x00000106ba 0x0000010d26 bearssl_synthetic v2 100 baseline > logs/baseline/bearssl_synthetic/v2/100/0xaa/launch_parse.log

./scripts/do_stats.sh 0xaa bearssl_synthetic v2 .90 .10 1 100 baseline > logs/baseline/bearssl_synthetic/v2/100/0xaa/launch_stats.log

### v3, 0xaa ###
mkdir -p logs/baseline/bearssl_synthetic/v3/100/0xaa; ./scripts/do_simulation.sh 0xaa bearssl_synthetic v3 100 baseline > logs/baseline/bearssl_synthetic/v3/100/0xaa/launch_simulation.log

./scripts/do_parse.sh 0xaa 0x00000105c8 0x00000105cc 0x00000109ce 0x00000106dc 0x00000109d2 bearssl_synthetic v3 100 baseline > logs/baseline/bearssl_synthetic/v3/100/0xaa/launch_parse.log 

./scripts/do_stats.sh 0xaa bearssl_synthetic v3 .90 .10 1 100 baseline > logs/baseline/bearssl_synthetic/v3/100/0xaa/launch_stats.log
