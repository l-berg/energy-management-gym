# Energy Management Gym

OpenAI Gym compatible environment that simulates the control of different
types of power plants by the agent. The reward depends on the total CO2 output of all
power plants and the deviation from the so-called residual load. The residual load is
the power output of renewable energy sources subtracted from the total load on the grid.
Data for all power grid and weather related information corresponds to the real
world data from Germany between 2015 and 2021.

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
python src/probing/smart_agent.py -a train -s 100000
```
and then evaluate the result after step 100000 by looking at sampled episodes with
```bash
python src/probing/smart_agent.py -a show -s 100000
```
To run pre-defined experiment number n, type
```bash
python src/probing/smart_agent.py -m auto -e <n> -a train
```
For a list of all options type
```bash
python src/probing/smart_agent.py --help
```

### Monitor training progress
Training may take some time and the final agent may not be the best one yet.
Use tensorboard to monitor the reward over time.
The current logging directory will be printed to the console, i.e.
```bash
Logging to tensorboard_logs/auto/1/42/PPO_0
```
To run tensorboard, type
```bash
tensorboard --logdir=<LOGDIR>
```
