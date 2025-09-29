cd /vepfs-dev/xing/workspace/DNA
export PYTHONPATH=$(pwd):$(pwd)/DQN
python=/vepfs-dev/xing/miniconda3/envs/dna/bin/python

#envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4'
envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 QbertNoFrameskip-v4 SeaquestNoFrameskip-v4 PongNoFrameskip-v4'
envs='DemonAttackNoFrameskip-v0'
declare -i i=0

for seed in $(seq 1 3)
do
#  echo ${seed}
  for env in ${envs}
  do
#    echo ${env}
    for wd in 1
    do
#      echo ${wd}
      for itv in 500 1000 2000
      do
        #export CUDA_VISIBLE_DEVICES=
        export CUDA_VISIBLE_DEVICES=$(($i%8))
        i+=1
#        misc="--enable_adams --enable_redo --adams_weight_decay ${wd} --redo_check_interval ${itv}"
#        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc} --exp_name  adams_wd_${wd}_redo_itv_${itv}  >>/dev/null 2>&1 &"

        misc="--enable_adam --enable_redo --redo_check_interval ${itv}"
        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc} --exp_name  adam_redo_itv_${itv}  >>/dev/null 2>&1 &"
        echo $cmd
        eval $cmd
      done
    done
  done
done
