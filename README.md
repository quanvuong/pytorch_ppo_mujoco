An implementation of PPO in Pytorch that achieves similar performance to the official implementation by OpenAI for the mujoco environments. I explained the implementation details [here](https://drive.google.com/file/d/1mjLKiPya9qSH9WuIO769fFGxgaTZT_qK/view?usp=sharing).

![](https://github.com/quanvuong/pytorch_ppo_mujoco/blob/master/images/Comparison_for_HalfCheetah-v2.png)

![](https://github.com/quanvuong/pytorch_ppo_mujoco/blob/master/images/Comparison_for_Hopper-v2.png)

![](https://github.com/quanvuong/pytorch_ppo_mujoco/blob/master/images/Comparison_for_InvertedDoublePendulum-v2.png)

![](https://github.com/quanvuong/pytorch_ppo_mujoco/blob/master/images/Comparison_for_InvertedPendulum-v2.png)

![](https://github.com/quanvuong/pytorch_ppo_mujoco/blob/master/images/Comparison_for_Reacher-v2.png)

![](https://github.com/quanvuong/pytorch_ppo_mujoco/blob/master/images/Comparison_for_Swimmer-v2.png)

![](https://github.com/quanvuong/pytorch_ppo_mujoco/blob/master/images/Comparison_for_Walker2d-v2.png)

Main Requirements:

- Pytorch

- OpenAI Gym

- Mujoco 150

More information about package requirements can be found in the conda `environment.yml` file

To start training, do:

`python main.py --env=InvertedPendulum-v2`
