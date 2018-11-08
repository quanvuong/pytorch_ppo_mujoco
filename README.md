An implementation of PPO in Pytorch that achieves similar performance to the official implementation by OpenAI for the mujoco environments. I explained the implementation details [here](https://drive.google.com/file/d/1mjLKiPya9qSH9WuIO769fFGxgaTZT_qK/view?usp=sharing) and [here](https://drive.google.com/file/d/1cWHWENpqBt9kgHoz5OlkH6bgMMhjnSBg/view?usp=sharing). I used the same hyperparameters as OpenAI's implementation to ensure apple-to-apple comparison.

Main Requirements:

- Pytorch

- OpenAI Gym

- Mujoco 150

More information about package requirements can be found in the conda `environment.yml` file

To start training, do:

`python main.py --env=InvertedPendulum-v2`
