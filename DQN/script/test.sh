cd /vepfs-dev/xing/workspace/DNA
export PYTHONPATH=$(pwd):$(pwd)/DQN
python=/vepfs-dev/xing/miniconda3/envs/dna/bin/python

#envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4'
envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 QbertNoFrameskip-v4 SeaquestNoFrameskip-v4 PongNoFrameskip-v4'
envs='DemonAttack-v4'
declare -i i=0

#misc=''
#
#for seed in $(seq 1 6)
#do
#    for env in ${envs}
#    do
#        #export CUDA_VISIBLE_DEVICES=
#        export CUDA_VISIBLE_DEVICES=$(($i%8))
#        i+=1
#        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc}  >>/dev/null 2>&1 &"
#        echo $cmd
#        eval $cmd
#
#    done
#done

#misc='--enable_redo'
#
#for seed in $(seq 1 6)
#do
#    for env in ${envs}
#    do
#        #export CUDA_VISIBLE_DEVICES=
#        export CUDA_VISIBLE_DEVICES=$(($i%8))
#        i+=1
#        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc}  >>/dev/null 2>&1 &"
#        echo $cmd
#        eval $cmd
#
#    done
#done

misc='--enable_muon'

for seed in $(seq 1 6)
do
    for env in ${envs}
    do
        #export CUDA_VISIBLE_DEVICES=
        export CUDA_VISIBLE_DEVICES=$(($i%8))
        i+=1
        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc}  >>/dev/null 2>&1 &"
        echo $cmd
        eval $cmd

    done
done
