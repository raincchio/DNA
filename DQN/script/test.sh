cd /vepfs-dev/xing/workspace/DNA
export PYTHONPATH=$(pwd):$(pwd)/DQN
python=/vepfs-dev/xing/miniconda3/envs/dna/bin/python

#envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4'
#envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 QbertNoFrameskip-v4 SeaquestNoFrameskip-v4 PongNoFrameskip-v4'
envs='AsterixNoFrameskip-v0'
#envs='BeamRiderNoFrameskip-v0 SpaceInvadersNoFrameskip-v0 AsterixNoFrameskip-v0 SeaquestNoFrameskip-v0 BreakoutNoFrameskip-v0 DemonAttackNoFrameskip-v0'
declare -i i=30
available_gpus=(0 1 2 3 4 5 6 7)

for seed in $(seq 1 6)
do
#  echo ${seed}
  for redo_tau in 0.1
  do
#    echo ${env}
    for eps in 1e-8
    do
#      echo ${wd}
      for env in $envs
      do
        export CUDA_VISIBLE_DEVICES=${available_gpus[$((i % ${#available_gpus[@]}))]}
        i+=1
        misc="--enable_adam"
#        misc="--enable_adam --enable_redo --adam_eps ${eps} --redo_tau ${redo_tau} --redo_check_interval ${itv}"
        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc} --exp_name  DQN_0107  >>/dev/null 2>&1 &"

#        misc="--enable_adam --enable_redo --redo_check_interval ${itv}"
#        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc} --exp_name  999_adam_redo_itv_${itv}  >>/dev/null 2>&1 &"
        echo $cmd
        eval $cmd
      done
    done
  done
done
