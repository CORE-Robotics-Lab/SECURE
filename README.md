# ShiElding with Control barrier fUnctions in inverse REinforcement learning (SECURE)

Codebase for SECURE

Implementations with Garage and Tensorflow.

Setup: Dependencies and Environment Preparation
---
The code is tested with Python 3.7 with Anaconda.

Required packages:
```bash
pip install numpy joblib==0.11 tensorflow scipy path PyMC3 cached-property pyprind gym==0.14.0 matplotlib dowel akro ray psutil setproctitle cma Box2D
```
  
[//]: # (```bash)

[//]: # (pip install numpy joblib==0.11 tensorflow-gpu==1.15.0 scipy path PyMC3 cached-property pyprind gym==0.14.0 matplotlib dowel akro ray psutil setproctitle cma Box2D)

[//]: # (```)

`gym==0.14.0` and `tensorflow-probability==0.8.0` does not like each other, so we need to separately install `tensorflow-probability`: 

```bash
pip install tensorflow-probability==0.8.0
```

If you are directly running python scripts, you will need to add the project root into your PYTHONPATH:
```bash
export PYTHONPATH=\path\to\this\repo\src
```

[//]: # (To use the Panda arm push domain, you need to install the panda-gym by following the [documentation]&#40;https://panda-gym.readthedocs.io/en/latest/&#41;.)
To use the Panda arm push domain, you need to install the panda-gym first. Run the following commands:
```shell
pip install -e panda-gym
```


Running SECURE
---

### Example - 1: Demolition derby

1) **Collect demonstrations and states:**
We have prepared the demonstrations and states for Demolition derby. 
Please find them in the following locations:
- Demonstrations: `src/demonstrations/demos_demolition_derby.pkl`
- States: `src/states/states_demolition_derby.pkl`

2) **Train CBF NN:**
The learned CBF NN is prepared at `data/demolition_derby/cbf_model`. 
You can also obtain it by:
```shell
python src/models/train_cbf_nn_demolition_derby.py
```


3) **Run the AIRL script on Demolition derby:**
The learned policy model is prepared at `data/demolition_derby/airl_model`.
You can also obtain it by:
```shell
# AIRL
python scripts/train_airl_demolition_derby.py
```

4) **Evaluate AIRL on Demolition derby:**
```shell
python scripts/evaluate_airl_demolition_derby.py
```
The results will be saved in `data/demolition_derby/airl_model/share/eval_results_just_airl.txt`.


5) **Evaluate SECURE on Demolition derby:**
```shell
python scripts/secure_demolition_derby.py
```
The results will be saved in `data/demolition_derby/airl_model/share/eval_results_secure.txt`.


### Example - 2: Panda arm push
1) **Collect demonstrations and states:**
We have prepared the demonstrations and states for Demolition derby. 
Please find them in the following locations:
- Demonstrations: `src/demonstrations/demos_panda_arm_push.pkl`
- States: `src/states/states_panda_arm_push.pkl`

2) **Train CBF NN:**
The learned CBF NN is prepared at `data/panda_arm_push/cbf_model`. 
You can also obtain it by:
```shell
python src/models/train_cbf_nn_panda_arm_push.py
```


3) **Run the AIRL script on Panda arm push:**
To learn a better AIRL policy for this more complex domain, we use BC for warmup. 
The learned policy model is prepared at `data/panda_arm_push/bc`.
You can also obtain it by:
```shell
# AIRL
python scripts/bc_airl_panda_arm_push.py
```

The learned policy model is prepared at `data/panda_arm_push/airl_model`.
You can also obtain it by:
```shell
# AIRL
python scripts/train_airl_panda_arm_push.py
```


4) **Evaluate AIRL on Panda arm push:**
```shell
python scripts/evaluate_airl_panda_arm_push.py
```
The results will be saved in `data/panda_arm_push/airl_model/share/eval_results_just_airl.txt`.


5) **Evaluate SECURE script on Panda arm push:**
```shell
python scripts/secure_panda_arm_push.py
```
The results will be saved in `data/panda_arm_push/airl_model/share/eval_results_secure.txt`.




### Other domains
1. Collect demonstrations to `src/demonstrations`.
2. Collect states to `src/states`.
2. Create a script modeled after `scripts/secure_demolition_derby.py`.
3. Change the environment, location, and log prefix. Use `circle()` if it's 2D environment, 
or use `fibonacci_sphere()` in `scripts/secure_panda_arm_push.py` if it's a 3D environment. 
4. Run the script analagous to demolition derby.

```
python scripts\{your_script}.py
```


Code Structure
---
The SECURE code is adjusted from the original [AIRL codebase](https://github.com/justinjfu/inverse_rl) and [MACBF](https://github.com/MIT-REALM/macbf).
