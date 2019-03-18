#!/usr/bin/env bash

for ((i=0;i<5;i+=1))
do
	python main.py \
	--env_name "HalfCheetah-v2" \
	--seed $i &

	python main.py \
	--env_name "Hopper-v2" \
	--seed $i &

	python main.py \
	--env_name "Walker2d-v2" \
	--seed $i &

	python main.py \
	--env_name "Ant-v2" \
	--seed $i &

	python main.py \
	--env_name "InvertedPendulum-v2" \
	--seed $i &

	python main.py \
	--env_name "InvertedDoublePendulum-v2" \
	--seed $i &

	python main.py \
	--env_name "Reacher-v2" \
	--seed $i &

	python main.py \
	--env_name "Humanoid-v2" \
	--seed $i &

	python main.py \
	--env_name "HumanoidStandup-v2" \
	--seed $i &

	python main.py \
	--env_name "Swimmer-v2" \
	--seed $i &
done
