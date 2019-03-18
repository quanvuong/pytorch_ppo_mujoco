#!/usr/bin/env bash

for ((i=0;i<5;i+=1))
do
	python main.py \
	--env "HalfCheetah-v2" \
	--seed $i &

	python main.py \
	--env "Hopper-v2" \
	--seed $i &

	python main.py \
	--env "Walker2d-v2" \
	--seed $i &

	python main.py \
	--env "Ant-v2" \
	--seed $i &

	python main.py \
	--env "InvertedPendulum-v2" \
	--seed $i &

	python main.py \
	--env "InvertedDoublePendulum-v2" \
	--seed $i &

	python main.py \
	--env "Reacher-v2" \
	--seed $i &

	python main.py \
	--env "Humanoid-v2" \
	--seed $i &

	python main.py \
	--env "HumanoidStandup-v2" \
	--seed $i &

	python main.py \
	--env "Swimmer-v2" \
	--seed $i &
done
