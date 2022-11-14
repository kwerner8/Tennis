## Learning algorithm

For this project, the Multi-Agent Deep Deterministic Policy Gradient (DDPG) algorithm was used to train the agent.

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

This approach is closely connected to Q-learning, and is motivated the same way.

Some characteristics of DDPG:
* DDPG is an off-policy algorithm.
* DDPG can only be used for environments with continuous action spaces.
* DDPG can be thought of as being deep Q-learning for continuous action spaces.

This approach extends DDPG to support multiple agents, where both agents are able to access
the observations and actions of each other and cooperate to keep the ball in play for as long as
possible.

## Model architecture and hyperparameters

The model architectures for the two neural networks used for the Actor and Critic are as follows:

Actor:
* Fully connected layer 1: Input 24 (state space of one agent), Output 256, RELU activation, Batch Normalization
* Fully connected layer 2: Input 256, Output 128, RELU activation
* Fully connected layer 3: Input 128, Output 2 (action space), TANH activation

Critic:
* Fully connected layer 1: Input 2*24 (state space of both agents), Output 256, RELU activation, Batch Normalization
* Fully connected layer 2: Input 2*24+2*2, Output 128, RELU activation, Input Layer1 and actions of both agents
* Fully connected layer 3: Input 128, Output 1

## Hyperparameters

```
BUFFER_SIZE = int(1e6)        # replay buffer size
BATCH_SIZE = 256              # minibatch size
GAMMA = 0.999                 # discount factor
TAU = 8e-3           	      # for soft update of target parameters
LR_ACTOR = 1e-4      	      # learning rate of the actor 
LR_CRITIC = 3e-4      	      # learning rate of the critic
WEIGHT_DECAY = 0.0    	      # L2 weight decay
LEARN_EVERY = 1       	      # learning timestep interval
LEARN_NUM = 5         	      # number of learning passes
START_LEARN = 500      	      # number of episodes before starting to learn
NOISE_REDUCTION_RATE = 0.99
NOISE_START=1.0
NOISE_END=0.01
```

## Results 

The current model achieved quite good results. The task was solved after 1105 episodes by reaching an average score of 0.5. The score continued to increase well above the required threshold. The agent was not trained for many more episodes, as shown in the graphic. This could lead to a possible deterioration of the average score, as described in the project description.

<img width="400" alt="Bildschirmfoto 2022-11-14 um 23 02 03" src="https://user-images.githubusercontent.com/100682506/201777913-5bc61c36-9279-4c11-85a1-6f24ad1c4d4b.png">



## Future work

* Use a shared critic for the two agents
* Explore distributed training using
	* A3C - Asynchronous Advantage Actor-Critic
	* A2C - Advantage Actor Critic




## Reference

1. https://spinningup.openai.com/en/latest/algorithms/ddpg.html
2. https://blog.openai.com/learning-to-cooperate-compete-and-communicate/
