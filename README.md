# Deep-Q-Network-in-Unity-3D-navigation-game
<img src='banana_collector.gif' width="500" height="300">

This repository holds the project code for using Deep Q-Network to learn to play a 3D navigation game - collecting banana in a 3D environment provided by Unity Technology. It is part of the Udacity [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) requirement. 

The state is represented as a 37 dimensional observations including the agent's velocity, ray-based perception of objects etc.. The agent has to learn to choose the optimal action based on the state it finds itself in. The action space is a 4 dimensional discrete space - 0 for forward, 1 for backward, 2 for turn left and 3 for turn right. The agent's goal is to collect as many as normal banana (1 point each) while avoiding purple banana (-1 point each); once the agent learns to collect an average of reward points of 13 and above the environment is considered solved.


## Installation
1. Install the Dependencies and setup python environment
Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning) to set up your Python environment.
2. Download the Unity Environment (specifically built for this project, **not** the Unity ML-Agents package). Then place the file in the root folder and unzip the file.
    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
3. Import the environment in Jupyter notebook under the the *drlnd* environment.
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
## How to Run
Load the Jupyter notebook *Report.ipynb* and run all cells.
