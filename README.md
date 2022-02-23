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

TODO: Write about policy

### 2. Data Collection

TODO: Write about data collection

### 3. Dynamics model

TODO: Write about dynamics model

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

If you want to use our perturbations, append argument `--perturbs-dir attacks/perturbations_ours` to above command. The
above command runs all the experiments shown in the paper. Add optional argument `--save` to save the figures, videos
in `results` folder. We already provided the results based on our perturbations in `results` folder.

### 6. Robustness experiment

TODO: Write about robustness experiment