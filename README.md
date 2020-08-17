# ***Resource Allocation in The Edge Computing Environment***

## Description

Considering the user mobility, offloading decision, and heterogeneous resources requirement of users in an edge computing environment, the problem is difficult to model and solve. The project aims to use reinforce learning model to allocate the resource in an edge computing environment.

In the simulation environment, fog servers provide computing resources for their clients to use and migrate tasks to appropriate servers to improve QoS according to the movement of mobile users. Determining the offloading server of each user is a discreet problem and allocating computing resources and migration bandwidth is a continuous problem. Thus, Deep Deterministic Policy Gradient (DDPG) can solve both problems and learns the optimal policy. The feature of updating model weight every step can adapt to a dynamic environment quickly.