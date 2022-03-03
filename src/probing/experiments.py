from datetime import timedelta

"""
 This file contains parameters for pre-set experiments

 features:
    always include:
        active plant
        plant output
        residual gen
    experiment 0:
        all features
        500000 steps
    experiment 1:
        weather data only
    experiment 2:
        residual load only
    experiment 3:
        time data only
    experiment 4:
        no weather, time or residual load
 actions:
    experiment 5:
        use absolute control mode
        num actions = 11
    experiment 6:
        relative control
        num actions = 17
        action range = 0.2

 scaling renewables:
    experiment 7:
        use 2030 data for wind/solar ~ 3x output

 time steps:
    experiment 8:
        use 5 minutes time intervals

 RL algorithms:
    experiment 9:
        use DQN
        500000 steps

 rewards:
    experiment 10:
        output diff scale 1
"""

EXPERIMENTS = [{
    # experiment 0 - use all features, train until convergence
    "algorithm": "PPO",
    "steps": 500000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
    }
}, {
    # experiment 1 - weather info only
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "use_residual_load": False,
        "use_weather_data": True,
        "use_time_data": False
    }
}, {
    # experiment 2 - residual load only
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "use_residual_load": True,
        "use_weather_data": False,
        "use_time_data": False
    }
}, {
    # experiment 3 - time info only
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "use_residual_load": False,
        "use_weather_data": False,
        "use_time_data": True
    }
}, {
    # experiment 4 - no weather, time or residual load info
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "use_residual_load": False,
        "use_weather_data": False,
        "use_time_data": False
    }
}, {
    # experiment 5 - use absolute control mode for power plants
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "relative_control": False,
        "num_actions": 11
    }
}, {
    # experiment 6 - larger action space
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "num_actions": 17,
        "action_range": 0.2
    }
}, {
    # experiment 7 - scale installed capacity of renewables(prediction for 2030)
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "solar_output_scale": 3,
        "wind_output_scale": 3
    }
}, {
    # experiment 8 - use 5 minute time intervalls for env step
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "step_period": timedelta(minutes=5)
    }
}, {
    # experiment 9 - use DQN instead of PPO, train until convergence
    "algorithm": "DQN",
    "steps": 500000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
    }
}, {
    # experiment 10 - change penalty for not hitting residual load
    "algorithm": "PPO",
    "steps": 150000,
    "sb3_params": {
        "policy": "MlpPolicy"
    },
    "env_params": {
        "output_diff_scale": 1
    }
}]
