export PYTHONPATH=$HOME/workspace/Redo/redo_ppo
python=$HOME/miniconda3/envs/redo/bin/python3

envs='HalfCheetah-v4'
task='ffn'
hids='64'
device='cpu'

for seed in $(seq 1 6)
do
    for env in ${envs}
    do
    	for hid in ${hids}
    	do
        	cmd="nohup ${python} -um main --seed=${seed} --device=${device} --env=${env} --hid=${hid} --task=${task} >/dev/null 2>&1 &"
        	echo $cmd
        	eval $cmd
        done
    done
done
