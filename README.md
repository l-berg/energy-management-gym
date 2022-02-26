# Energy Management Gym

## Installation
We use [anaconda](https://www.anaconda.com/products/individual) to manage dependencies.
```bash
conda env create -f environment.yml  # may take a while
conda activate energy-gym
export PYTHONPATH=.
```

Now run the scripts from the projects root directory, e.g.:
```bash
python src/probing/random_agent.py
```


## Agents
To get a general idea of the environment, take a look at what the random agent does:
```bash
python src/probing/random_agent.py
```

### Train and evaluate
To probe the environment, first train an agent for 100000 steps with
```bash
python src/probing/smart_agent.py train 100000
```
and then evaluate the result after step 100000 by looking at sampled episodes with
```bash
python src/probing/smart_agent.py show 100000
```

### Monitor training progress
Training may take some time and the final agent may not be the best one yet.
Use tensorboard to monitor the reward over time:
```bash
tensorboard --logdir=tensorboard_logs
```
