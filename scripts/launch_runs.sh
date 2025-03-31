#SET THESE VARIABLES.
PASSWD_FILE='passwd_file'
USER='barberk'

SIM_ROOT="$(pwd)"

export SIM_ROOT

keys=('0xaa' '0x44' 'rand-0.10_0.90' 'rand-0.20_0.80' 'rand-0.30_0.70' 'rand-0.40_0.60' 'rand-0.50_0.50' 'rand-0.60_0.40' 'rand-0.70_0.30' 'rand-0.80_0.20' 'rand-0.90_0.10') 
#keys=('window-4bit-v1' 'window-4bit-v2' 'window-4bit-v3' 'window-4bit-v4' 'window-4bit-v5' 'window-4bit-v6' 'window-4bit-v7' 'window-4bit-v8' 'window-4bit-v9' 'window-4bit-v10')
#'rand-0.10_0.90' 'rand-0.10_0.90.v1' 'rand-0.10_0.90.v2' 'rand-0.10_0.90.v3' 'rand-0.10_0.90.v4' 'rand-0.10_0.90.v5' 'rand-0.10_0.90.v6' 'rand-0.10_0.90.v7' 'rand-0.10_0.90.v8' 'rand-0.10_0.90.v9'
#'rand-0.20_0.80' 'rand-0.20_0.80.v1' 'rand-0.20_0.80.v2' 'rand-0.20_0.80.v3' 'rand-0.20_0.80.v4' 'rand-0.20_0.80.v5' 'rand-0.20_0.80.v6' 'rand-0.20_0.80.v7' 'rand-0.20_0.80.v8' 'rand-0.20_0.80.v9'
#'rand-0.30_0.70' 'rand-0.30_0.70.v1' 'rand-0.30_0.70.v2' 'rand-0.30_0.70.v3' 'rand-0.30_0.70.v4' 'rand-0.30_0.70.v5' 'rand-0.30_0.70.v6' 'rand-0.30_0.70.v7' 'rand-0.30_0.70.v8' 'rand-0.30_0.70.v9' 
#keys=('rand-0.40_0.60' 'rand-0.40_0.60.v1' 'rand-0.40_0.60.v2' 'rand-0.40_0.60.v3' 'rand-0.40_0.60.v4' 'rand-0.40_0.60.v5' 'rand-0.40_0.60.v6' 'rand-0.40_0.60.v7' 'rand-0.40_0.60.v8' 'rand-0.40_0.60.v9')
#keys=('rand-0.50_0.50' 'rand-0.50_0.50.v1' 'rand-0.50_0.50.v2' 'rand-0.50_0.50.v3' 'rand-0.50_0.50.v4' 'rand-0.50_0.50.v5' 'rand-0.50_0.50.v6' 'rand-0.50_0.50.v7' 'rand-0.50_0.50.v8' 'rand-0.50_0.50.v9')
#'rand-0.60_0.40' 'rand-0.60_0.40.v1' 'rand-0.60_0.40.v2' 'rand-0.60_0.40.v3' 'rand-0.60_0.40.v4' 'rand-0.60_0.40.v5' 'rand-0.60_0.40.v6' 'rand-0.60_0.40.v7' 'rand-0.60_0.40.v8' 'rand-0.60_0.40.v9'
#keys=('rand-0.70_0.30' 'rand-0.70_0.30.v1' 'rand-0.70_0.30.v2' 'rand-0.70_0.30.v3' 'rand-0.70_0.30.v4' 'rand-0.70_0.30.v5' 'rand-0.70_0.30.v6' 'rand-0.70_0.30.v7' 'rand-0.70_0.30.v8' 'rand-0.70_0.30.v9')
#'rand-0.80_0.20' 'rand-0.80_0.20.v1' 'rand-0.80_0.20.v2' 'rand-0.80_0.20.v3' 'rand-0.80_0.20.v4' 'rand-0.80_0.20.v5' 'rand-0.80_0.20.v6' 'rand-0.80_0.20.v7' 'rand-0.80_0.20.v8' 'rand-0.80_0.20.v9'
#'rand-0.90_0.10' 'rand-0.90_0.10.v1' 'rand-0.90_0.10.v2' 'rand-0.90_0.10.v3' 'rand-0.90_0.10.v4' 'rand-0.90_0.10.v5' 'rand-0.90_0.10.v6' 'rand-0.90_0.10.v7' 'rand-0.90_0.10.v8' 'rand-0.90_0.10.v9' 
action='stats'
mode='local'
suite='bearssl'
apps=('v1' 'v2' 'v3')

snode=4

excluded_nodes=('10' '11' '12')

phi=".90"
alpha=".10"
window="1"
design="baseline"

iters=100

echo "#####################"
while [ "$1" != '' ]; do
	case $1 in
		-action )
			shift
			action=$1
			echo "action: "$action
			;;
                -mode )
                        shift
                        mode=$1
                        echo "mode: "$mode
                        ;;  
		-keysi )
			shift
			keys=($1)
			;;
		-suite )
			shift
			suite=$1
			;;
		-appsi )
			shift
			apps=($1)
			;;
		-iters )
			shift
			iters=$1
			;;
		-snode )
			shift
			snode=$1
			echo "start arch node: "$snode""
			;;
		-phi )
			shift
			phi=$1
			;;
		-alpha )
			shift
			alpha=$1
			;;
		-window )
			shift
			window=$1
			;;
		-design )
			shift
			design=$1
			;;
	esac
	shift
done
echo "#####################"

node=$snode
command=""

if [ "$action" == "kill" ]; then
        echo "Killing all processes..."
        while [ "$node" -lt 13 ]; do
                echo "node $node"
                command="pkill -u "$USER
                if [ "$mode" == "ssh" ]; then
 			sshpass -f "$PASSWD_FILE" ssh -o StrictHostKeyChecking=no "$USER"@arch$node.cse.ohio-state.edu "$command"
		elif [ "$mode" == "dryrun" ]; then
			echo $command
		fi
                let node++
                until [[ ! " ${excluded_nodes[@]} " =~ " ${node} " ]]
                do
                    let node++
                done
        done
        exit 1
fi

for app in "${apps[@]}"
do
	for key in "${keys[@]}"
	do
                let node++
		until [[ ! " ${excluded_nodes[@]} " =~ " ${node} " ]]
		do
			let node++
		done
		if [ "$node" -gt 12 ]; then
			node=$snode
		fi

		command+="cd "$PWD"; "
	
		if [ "$action" == "simulate" ]; then
			command+="mkdir -p logs/"$design"/"$suite"/"$app"/"$iters"/"$key"; nohup ./scripts/do_simulation.sh "$key" "$suite" "$app" "$iters" "$design" > logs/"$design"/"$suite"/"$app"/"$iters"/"$key"/launch_simulation.log 2>&1 &"
		
		elif [ "$action" == "parse" ]; then
			command+="nohup ./scripts/do_parse.sh "$key 
			if [ "$suite" == "microbench" ] && [ "$app" == "ct_ccopy" ]; then
				command+=" 0x008000010e 0x0080000124 0x0080000196 0x0080000130 0x008000019a "
                	elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v1" ]; then
				command+=" 0x0000010128 0x000001012c 0x00000106d2 0x000001022c 0x00000106d6 "
                	elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v1_warmup" ]; then
				command+=" 0x000001014a 0x000001014e 0x00000106f4 0x000001024e 0x00000106f8 "
                	elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v1_fence" ]; then
				command+=" 0x0000010128 0x000001012c 0x00000107a4 0x000001022c 0x00000107a8 "
			elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v2" ]; then
				command+=" 0x000001012c 0x0000010130 0x0000010882 0x000001021a 0x0000010886 "
			elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v2_warmup" ]; then
				command+=" 0x0000010152 0x0000010156 0x00000108a8 0x0000010240 0x00000108ac "
			elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v2_fence" ]; then
				command+=" 0x000001012c 0x0000010130 0x000001095c 0x000001021a 0x0000010960 "
			elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v3" ]; then
				command+=" 0x0000010128 0x000001012c 0x000001052e 0x000001023c 0x0000010532 "
			elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v3_warmup" ]; then
				command+=" 0x000001014a 0x000001014e 0x0000010550 0x000001025e 0x0000010554 "
			elif [ "$suite" == "bearssl_synthetic" ] && [ "$app" == "v3_fence" ]; then
				command+=" 0x0000010128 0x000001012c 0x0000010600 0x000001023c 0x0000010604 "
			fi

			command+=$suite" "$app" "$iters" "$design" > logs/"$design"/"$suite"/"$app"/"$iters"/"$key"/launch_parse.log 2>&1 &"

		elif [ "$action" == "stats" ]; then
			command+="nohup ./scripts/do_stats.sh "$key" "$suite" "$app" "$phi" "$alpha" "$window" "$iters" "$design" > logs/"$design"/"$suite"/"$app"/"$iters"/"$key"/launch_stats.log 2>&1 &"
		fi

		if [ "$mode" == "ssh" ]; then
			echo "Launching "$design":"$app":"$key" on arch"$node""
			sshpass -f "$PASSWD_FILE" ssh -o StrictHostKeyChecking=no "$USER"@arch$node.cse.ohio-state.edu "$command"
		elif [ "$mode" == "dryrun" ]; then
			echo "Launching "$design":"$app":"$key""
			echo $command
		elif [ "$mode" == "local" ]; then
			eval $command
		fi

		command=""
	done
done
