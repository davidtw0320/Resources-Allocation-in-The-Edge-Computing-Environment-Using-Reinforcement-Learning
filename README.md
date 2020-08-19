# ***Resource Allocation in The Edge Computing Environment***

## Summary
With the development of cloud computing based mobile applications, such as object detection, face recognition, have become overwhelming in recent years. However, because of the remote execution, cloud computing may cause high latency and increase the backhaul bandwidth consumption. To address this problem, edge computing is promising to improve response times and relieve the backhaul pressure by moving the storage and computing resources closer to mobile users.

Considering the mobile user mobility, offloading decision, and heterogeneous resources requirement in an edge computing environment, the project aims to use Reinforce Learning (RL) model to allocate the resource in an edge computing environment.

 ![gui](image/summary.png)
 picture originated from: [IEEE Inovation at Work](https://innovationatwork.ieee.org/real-life-edge-computing-use-cases/)
***

## Prerequisite

+ Python 3.7.5
+ Tensorflow 2.2.0
+ Tkinter 8.6

***

## Build Setup

### *Run The System*

```cmd
# execute
$ python3 src/run_this.py
```

### *Text Interface Eable / Diable* (in run_this.py)

```python
TEXT_RENDER = True / False
```

### *Graphic Interface Eable / Diable* (in run_this.py)

```python
SCREEN_RENDER = True / False
```

***

## Key Point

## *Edge Computing Environment*

+ Mobile User
  + Users move according to the mobility data provided by [CRAWDAD](https://crawdad.org/index.html). This data was collected from the users of mobile devices at the subway station.
  + User's device offload tasks to one edge server.
  + After the request task has been processed, users need to receive processed packets from the edge server and send a new task to the edge server again.

+ Edge Server
  + Responsible for offering computational resources *(6.3 * 1e7 byte/sec)* and processing tasks for mobile users.
  + Each edge server can only offer service to limited numbers of users and allocate computational resources to them.
  + The task may be migrated from one edge server to another one within limited bandwidth *(1e9 byte/sec)*.

+ Request Task: [VOC SSD300 Objection detection](hhttps://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)
  + state 1 : start to offload a task to the edge server
  + state 2 : request task is on the way to the edge server *(2.7 * 1e4 byte)*
  + state 3 : request task is proccessed *(1.08 * 1e6 byte)*
  + state 4 : request task is on the way back to the mobile user *(96 byte)*
  + state 5 : disconnect (default)
  + state 6 : request task is migrated to another edge server

+ Graphic Interface

  ![gui](image/graphic_interface.png)
  + Edge servers *(static)*
    + Big dots with consistent color
  + Mobile users *(dynamic)*
    + Small dots with changing color
    + Color
      + the color of edge servers : request task is handled by the edge server with the same color and in state 1 ~ state 4  
      + Red : request task is in state 5
      + Green : request task is in state 6

## *Deep Deterministic Policy Gradient* (in DDPG.py)

+ Description
  
  While determining the offloading server of each user is a discrete variable problem, allocating computing resources and migration bandwidth are continuous variable problems. Thus, Deep Deterministic Policy Gradient (DDPG), a model-free off-policy actor-critic algorithm, can solve both discrete and continuous problems. Also, DDPG updates model weights every step, which means the model can adapt to a dynamic environment instantly.

+ State

  ```python
    def generate_state(two_table, U, E, x_min, y_min):
        one_table = two_to_one(two_table)
        S = np.zeros((len(E) + one_table.size + len(U) + len(U)*2))
        count = 0
        for edge in E:
            S[count] = edge.capability/(r_bound*10)
            count += 1
        for i in range(len(one_table)):
            S[count] = one_table[i]/(b_bound*10)
            count += 1
        for user in U:
            S[count] = user.req.edge_id/100
            count += 1
        for user in U:
            S[count] = (user.loc[0][0] + abs(x_min))/1e5
            S[count+1] = (user.loc[0][1] + abs(y_min))/1e5
            count += 2
        return S
  ```

  + **Available computing resource** of each edge server
  + **Available migration bandwidth** of each connection between edge servers
  + **Offloading target** of each mobile user
  + **Location** of each mobile user

+ Action

  ```python
  def generate_action(R, B, O):
    a = np.zeros(USER_NUM + USER_NUM + EDGE_NUM * USER_NUM)
    a[:USER_NUM] = R / r_bound
    # bandwidth
    a[USER_NUM:USER_NUM + USER_NUM] = B / b_bound
    # offload
    base = USER_NUM + USER_NUM
    for user_id in range(USER_NUM):
        a[base + int(O[user_id])] = 1
        base += EDGE_NUM
    return a
  ```

  + **Computing resource** allocated to each mobile user's task (continuous)
  + **Migration bandwidth** of each mobile user's task needs to occupy (continuous)
  + **offloading target** of each mobile user (discrete)

+ Reward
  + total processed tasks in each step

+ Model Architecture

  ![ddpg architecture](image/DDPG_architecture.png)

***

## Simulation Result

+ Simulation Environment
  + 10 edge servers with computational resource *(6.3 * 1e7 byte/sec)*
  + Each edge server can offer at most 4 task processing services.
  + 3000 steps/episode, 90000 sec/episode

+ Result
    | Number of Clients | Average Total proccessed tasks in the last 10 episodes| Training History |
    | ------- | -------- | -------- |
    | 10 | 11910 | ![result](output/ddpg_10u10e4lKAIST/rewards.png) |
    | 20 | 23449 | ![result](output/ddpg_20u10e4lKAIST/rewards.png) |
    | 30 | 33257 | ![result](output/ddpg_30u10e4lKAIST/rewards.png) |
    | 40 | 40584 | ![result](output/ddpg_40u10e4lKAIST/rewards.png) |

***

## Reference

+ Mobility Data
  
  [Mongnam Han, Youngseok Lee, Sue B. Moon, Keon Jang, Dooyoung Lee, CRAWDAD dataset kaist/wibro (v. 2008‑06‑04), downloaded from https://crawdad.org/kaist/wibro/20080604, https://doi.org/10.15783/C72S3B, Jun 2008.](https://crawdad.org/kaist/wibro/20080604)
