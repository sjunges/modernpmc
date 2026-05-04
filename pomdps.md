---
numbering:
  heading_1: true
  heading_2: true
  heading_3: true
  equations: false  

kernelspec:
  name: python3
  display_name: Python 3
---

# Partially observable MDPs
In this chapter, we discuss POMDPs.

## What are POMDPs?

```{code-cell} python
:tags: [remove-input]
import stormpy
import sympy
import stormvogel as sv
import stormvogel.teaching as teach
import stormvogel.bird as bird
from stormvogel.to_dot import plot_model_pydot
sympy.init_printing()
from IPython.display import Math
```

````{prf:example}
Consider the Monty-hall problem: 
You play in a game show, there are three doors, 
behind one of these doors is a car, behind the other doors are goats.
The assumption here is that you want to win the car.
You must pick a door, and without any information here, any pick is equally good.
Now the game master interfers: 
They reveal that after a randomly picked door different then the one you picked is a goat.
The game master now asks you whether you want to stick to your choice or change.
The simple question is? Should you change.
To answer this question, you may be tempted to construct the following MDP:
```{code-cell} python
from stormvogel.examples import create_monty_hall_mdp
#TODO add montyhall MDP here
monty_hall_mdp = create_monty_hall_mdp()
plot_model_pydot(monty_hall_mdp)
```
Let's look at the optimal probability to win (and the policy that is optimal):
```
```
````
The problem in the example above is that the agent (the player in the game show) can make decisions based on unavailable information.
This analysis is thus inadequate for any purposes where you want to prove that a sensor-based system can achieve a goal or that an attacker cannot steal sensitive information.
To overcome these limitations, POMDPs have been introduced. They model explicitly that at every state, only limited information is available.
````{prf:example}
```{code-cell} python
from stormvogel.examples import create_monty_hall_pomdp
#TODO add montyhall MDP here
monty_hall_mdp = create_monty_hall_pomdp()
plot_model_pydot(monty_hall_mdp)
```
````

```{prf:definition}
A _partially observable MDP_ (POMDP) is a triple $\langle \mdp, \Obs, \obsfun \rangle$, 
where $\mdp$ is an MDP with states $S$, $\Obs$ is a finite nonempty set of _observations_,
and $\obsfun\colon S \rightarrow \Distr{\Obs}$ is a _state-based observation function_.
```
```{prf:remark}
Beyond state-based observations, the literature also uses transition-based observations where information is shared when taking a transition.
We do not use such models here.
```
```{warning} Assumption
From here on, assume deterministic observations with the shape $\obsfun\colon S \rightarrow \Obs$.
Any stochastic observation function can be transformed into a determinsitic one using a polynomial blowup.
```
```{prf:definition} Observation-based policies

```
Indeed, 


# Algorithms for POMDPs 
## 

## Optimising over memoryless, randomizing policies



