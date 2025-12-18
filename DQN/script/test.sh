cd /vepfs-dev/xing/workspace/DNA
export PYTHONPATH=$(pwd):$(pwd)/DQN
python=/vepfs-dev/xing/miniconda3/envs/dna/bin/python

#envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4'
#envs='BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 EnduroNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 QbertNoFrameskip-v4 SeaquestNoFrameskip-v4 PongNoFrameskip-v4'
envs='DemonAttackNoFrameskip-v0'
#envs='BeamRiderNoFrameskip-v0 SpaceInvadersNoFrameskip-v0 AsterixNoFrameskip-v0 SeaquestNoFrameskip-v0'
declare -i i=6
available_gpus=(0 3 4 5 6)

for seed in $(seq 1 6)
do
#  echo ${seed}
  for redo_tau in 0.05 0.2
  do
#    echo ${env}
    for eps in 1e-9
    do
#      echo ${wd}
      for itv in 1000
      do
        export CUDA_VISIBLE_DEVICES=${available_gpus[$((i % ${#available_gpus[@]}))]}
        i+=1
        misc="--enable_adam --enable_redo --adam_eps ${eps} --redo_tau ${redo_tau} --redo_check_interval ${itv}"
        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${envs} ${misc} --exp_name  test_eps_${eps}_tau_${redo_tau}_dqn_redo  >>/dev/null 2>&1 &"

#        misc="--enable_adam --enable_redo --redo_check_interval ${itv}"
#        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc} --exp_name  999_adam_redo_itv_${itv}  >>/dev/null 2>&1 &"
        echo $cmd
        eval $cmd
      done
    done
  done
done
