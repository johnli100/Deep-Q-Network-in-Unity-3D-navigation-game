# Deep-Q-Network-in-Unity-3D-navigation-game
<img src='banana_collector.gif' width="500" height="300">

This repository holds the project code for using Deep Q-Network to learn to play a 3D navigation game - collecting banana in a 3D environment provided by Unity Technology. It is part of the Udacity [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) requirement. The end result is a trained agent capable of collecting as many rewards as possible (by collecting yellow banana and avoiding purple ones).

## Installation
1. Create (and activate) a new environment with Python 3.6
  * **Linux** or **Mac**:
```
conda create --name drlnd python=3.6
source activate drlnd
```
  * **Windows**:
```
conda create --name drlnd python=3.6 
activate drlnd
```
2. Create an IPython kernel for the drlnd environment and choose *drlnd* environment in Jupyter notebook under *Kernel>Change Kernel* 
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
3. Download the Unity Environment (specifically built for this project, **not** the Unity ML-Agents package). Then place the file in the root folder and unzip the file.
    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
4. Import the environment in Jupyter notebook under the the *drlnd* environment.
```
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="[to be replaced with file below depending on OS]")
```
Replace the file name with the following depending on OS:  
  * Mac: "Banana.app"
  * Windows (x86): "Banana_Windows_x86/Banana.exe"
  * Windows (x86_64): "Banana_Windows_x86_64/Banana.exe"
  * Linux (x86): "Banana_Linux/Banana.x86"
  * Linux (x86_64): "Banana_Linux/Banana.x86_64"
  * Linux (x86, headless): "Banana_Linux_NoVis/Banana.x86"
  * Linux (x86_64, headless): "Banana_Linux_NoVis/Banana.x86_64"

