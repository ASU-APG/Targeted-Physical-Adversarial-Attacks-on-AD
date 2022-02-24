# Targeted-Physical-Adversarial-Attacks-on-AD

Targeted Attack on Deep RL-based Autonomous Driving with Learned Visual Patterns

https://arxiv.org/abs/2109.07723 (Accepted at [ICRA 2022](https://www.icra2022.org))

## Overview

The end to end implementation of paper follows a series of below steps

1. Obtaining a policy
2. Collecting data from simulator
3. Training dynamics model
4. Optimizing the perturbation for physical adversarial attack
5. Testing the physical adversarial attack
6. Evaluating robustness of attack to object position

### 1. Policy

We used a policy obtained using Actor-Critic algorithm for which the pretrained agent is taken from
this [repo](https://github.com/xtma/pytorch_car_caring) and modified a bit for our use.

### 2. Data Collection

We collect data by running the agent with pretrained + noise policy as explained in the paper. This is done using below
commands for all three driving scenarios.

#### Scenario - Straight

```commandline
python data_collection/generation_script.py --same-track --rollouts 1 --rootdir datasets --policy pre --scenario straight
python data_collection/generation_script.py --same-track --rollouts 9999 --rootdir datasets --policy pre_noise --scenario straight
```

#### Scenario - Left Turn

```commandline
python data_collection/generation_script.py --same-track --rollouts 1 --rootdir datasets --policy pre --scenario left_turn
python data_collection/generation_script.py --same-track --rollouts 9999 --rootdir datasets --policy pre_noise --scenario left_turn
```

#### Scenario - Right Turn

```commandline
python data_collection/generation_script.py --same-track --rollouts 1 --rootdir datasets --policy pre --scenario right_turn
python data_collection/generation_script.py --same-track --rollouts 9999 --rootdir datasets --policy pre_noise --scenario right_turn
```

**NOTE**: The data collection scripts provide multiple options to get more datasets with different policy types if
wanted.

### 3. Dynamics model

Dynamics model is trained using two models (VAE and MDRNN)

#### VAE

```commandline
python dynamics_model/trainvae.py --dataset scenario_straight
```

#### MDRNN

```commandline
python dynamics_model/trainmdrnn.py --dataset scenario_straight
```

**NOTE**: Change `--dataset` argument to `scenario_left_turn` and `scenario_right_turn` for other two driving scenarios
respectively.

### 4. Generate Adversarial Perturbations

```commandline
python attacks/optimize.py --scenario straight
```

To optimize for other scenarios, change scenario argument to `left_turn` and
`right_turn` respectively.

**TIP**: use `--help` to know available arguments and play around with different time steps, perturbation strength etc.

The perturbations are by default saved in `attacks/perturbations` folder segregated by each driving scenario. Further,
we are providing our optimized perturbations shown in paper under `attacks/perturbations_ours` folder to easily allow
for testing in next step.

### 5. Test Physical Adversarial Attack

```commandline
python attacks/test.py --scenario straight
```

If you want to use our perturbations, append argument `--perturbs-dir attacks/perturbations_ours` to the command. The
above command runs all the experiments shown in the paper. Add optional argument `--save` to save the figures, videos
in `results` folder. We already provided the results based on our perturbations in `results` folder.

### 6. Robustness experiment

```commandline
python attacks/robustness.py --scenario straight
```

If you want to use our perturbations, append argument `--perturbs-dir attacks/perturbations_ours` to the command. The
above command runs the robustness experiment shown in the paper. Add optional argument `--save` to save the robustness
heatmap in `results` folder. We already provided the robustness result based on our perturbations in `results` folder.

### Acknowledgement

We would like to thank developers of below open source code for providing policy and dynamics model implementations
which are used in our code.

- Policy - https://github.com/xtma/pytorch_car_caring
- Dynamics model - https://github.com/ctallec/world-models
