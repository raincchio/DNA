export PYTHONPATH=/vepfs-dev/xing/workspace/redo_dqn
python=/vepfs-dev/xing/miniconda3/envs/ddqn/bin/python

#envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4'
envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 QbertNoFrameskip-v4 SeaquestNoFrameskip-v4 PongNoFrameskip-v4'
envs='DemonAttackNoFrameskip-v4'
declare -i i=0

for seed in $(seq 1 6)
do
    for env in ${envs}
    do
        #export CUDA_VISIBLE_DEVICES=
        export CUDA_VISIBLE_DEVICES=$(($i%8))
        i+=1
        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} --enable_redo  >>/dev/null 2>&1 &"
        echo $cmd
        eval $cmd

    done
done

#for seed in $(seq 1 6)
#do
#    for env in ${envs}
#    do
#        #export CUDA_VISIBLE_DEVICES=
#        export CUDA_VISIBLE_DEVICES=$(($i%8))
#        i+=1
#        cmd="nohup ${python} -um main --seed=${seed} --env_id=${env} --mrt  --mrt_interval 2000 >>/dev/null 2>&1 &"
#        echo $cmd
#        eval $cmd
#
#    done
#done
#
#for seed in $(seq 1 6)
#do
#    for env in ${envs}
#    do
#        #export CUDA_VISIBLE_DEVICES=
#        export CUDA_VISIBLE_DEVICES=$(($i%8))
#        i+=1
#        cmd="nohup ${python} -um main --seed=${seed} --env_id=${env} >>/dev/null 2>&1 &"
#        echo $cmd
#        eval $cmd
#
#    done
#done

