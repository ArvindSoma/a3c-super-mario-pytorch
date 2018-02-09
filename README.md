# Reinforcement Learning for Super Mario Bros using A3C on GPU

This project is based on the implementation of the paper [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) using custom training modifications. This project was created for the course [Deep Learning for Computer Vision](https://vision.in.tum.de/teaching/ws2017/dl4cv) held at TUM.

## Prerequisites
- Python3.5+
- Pytorch
- OpenAI Gym

## Getting Started
Install the following packages using the given commands
```
sudo apt-get update
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo apt-get install fceux
```

Now Super Mario Bros NES environment has to be set up. We are going to use [Philip Paquette's Super Mario Bros](https://github.com/ppaquette/gym-super-mario) implementation for gym with some modifications to run on the current OpenAI Gym version.
Follow [Issue 6](https://github.com/ppaquette/gym-super-mario/issues/6) to get the Mario NES environment up and running.

To match the default settings of this project modify the ''gym/envs/__init__.py'' to register env
```
register(
     id='metaSuperMarioBros-1-1-v0',
     entry_point='gym.envs.ppaquette_gym_super_mario:MetaSuperMarioBrosEnv',
)
```
No matter what 'id' is set to, use the MetaSuperMarioBrosEnv entry point to remove frequent closing of the emulator.

## Training and Testing
To train the network from scratch, use the following command
```
python3 train-mario.py --num-processes 8
```

This command requires atleast an 8-Core system with atleast 16GB memory and 8GB GPU-RAM. 
You can reduce the number of processes to run on a personal system, but expect the training time to increase drastically.
```
python3 train-mario.py --num-processes 2 --non-sample 1
```
The program works on random and non-random processes so that the training converges faster. By default there are two non-random processes whcih can be changed using args.
1 test process is created with remaining train processes. Test stores data in a CSV file inside save folder, which can be plotted later. 

More arguments are mentioned in the file.

## Results
After ~20 hours of training on 8 processes (7 Train, 1 Test) the game converges.

![](video/mario-level1.gif)

## Repository References
This project heavily relied on [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c).

