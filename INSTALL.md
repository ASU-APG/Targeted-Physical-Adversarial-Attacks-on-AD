# How to setup the Environment
The installation below has been tested on Linux and MacOS computers. We were unfortunately not able to test the setup on Windows computers but are willing to help debug setups.

**Please make sure to add the current repository to the PYTHONPATH**
```
export PYTHONPATH=<repo-dir>
```
# For MacOS and Linux
It is recommended to use conda environment to run our code. You can create a conda environment with python 3.8 through the following:
```
conda create --name advenv python == 3.8
conda activate advenv 
```

Before installing the packages for the code, please download swig through conda:
```
conda install swig -y
```
We can now install the packages necessary to run the code:
```
pip install -r requirements.txt
```

Due to this [issue](https://github.com/openai/gym/issues/2101) for MacOS BigSur and later, we have decided to not include pyglet in the `requirements.txt` file and separately install it: 
```
pip install pyglet==1.5.11
```
**Note:** There will be an error regarding pip's dependency resolver that pops up after installing, but pyglet will still be successfully installed and the code will still be able to run.

For some reason unknown to us, the following error pops up: 
```
AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'
```
The solution that fixed the problem was to run `pip install box2d-py`; however, including box2d-py in the `requirements.txt` file produces this error again. We found that manually calling the pip install fixes the problem. If anyone understands why this error occurs, please let us know either through email ([tz98@cornell.edu](tz98@cornell.edu) & [pbuddare@asu.edu](pbuddare@asu.edu)) or through a comment in this repository. Please install:
```
pip install box2d-py
```

# Setting up Custom Gym installation
In order to add our custom environment to gym, run the following code: 
```
cd gym
pip install -e .
```