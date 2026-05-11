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

# What are POMDPs?

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
from fractions import Fraction
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
:tags: [hide-input]
from stormvogel.examples import create_condensed_monty_hall
monty_hall_mdp = create_condensed_monty_hall().copy().make_fully_observable()
plot_model_pydot(monty_hall_mdp)
```
Let's look at the optimal probability to win (and the policy that is optimal):
```{code-cell} python
:tags: [hide-input]
result = sv.model_checking(monty_hall_mdp, 'Pmax=? [F "win"]')
print(f"Optimal win probability: {result.at_init()}")
# TODO: visualise scheduler
```
The fully observable MDP admits a policy that wins with probability 1: stay at $g'$ (the initial pick was correct) and switch at $b'$ (the initial pick was wrong).
This policy, however, relies on distinguishing states $g'$ and $b'$ — that is, knowing whether the initial door choice was correct.
In the actual game, the player has no access to this information.
````
The problem in the example above is that the agent (the player in the game show) can make decisions based on unavailable information.
This analysis is thus inadequate for any purposes where you want to prove that a sensor-based system can achieve a goal or that an attacker cannot steal sensitive information.
To overcome these limitations, POMDPs have been introduced. They model explicitly that at every state, only limited information is available.
````{prf:example}
In the POMDP, observations hide the car's location.
States $d_1$, $d_2$, $d_3$ share observation `pick?`: the player knows they must pick a door but cannot see where the car is.
States $g$ and $b$ share observation `switch?`: after picking, the player cannot tell whether they chose correctly.
States $g'$ and $b'$ share observation `show!`: after the host reveals a goat, the player still cannot tell whether their original pick was right.

Observation-based policies can therefore only act on these three observation phases.
At observation `pick?` the player chooses one of three doors; at observation `show!` they choose to stay or switch.
This gives $3 \times 2 = 6$ deterministic observation-based policies (and their randomised convex combinations).
```{code-cell} python
:tags: [hide-input]
from stormvogel.examples import create_condensed_monty_hall
monty_hall_pomdp = create_condensed_monty_hall()
plot_model_pydot(monty_hall_pomdp)
```
````

```{prf:definition} Partially Observable MDP
:label: def:pomdp
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
Given the assumption on deterministic observations, we write $$ S_z = \{ s \in S \mid \obsfun(s) = z \} $$

```{warning} Assumption
For all $s, s' \in S_z$, we assume $\EnAct{s} = \EnAct{s'}$.
We then write $\EnAct{z}$ for the common set of enabled actions at observation $z$.
```

```{prf:definition} Observation trace
:label: def:pomdp:obstrace
Given any path $$ \xi = s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} \dots $$
the observation trace is $$ \obstr{\xi} = \obsfun(s_0) \xrightarrow{a_0} \obsfun(s_1) \xrightarrow{a_1} \obsfun(s_2) \xrightarrow{a_2} \dots  $$
```
```{prf:definition} Observation-based policies
:label: def:pomdp:obspolicies
A policy $\pi$ is _observation-based_, if for $\xi, \xi' \in \Paths$ 
$$
\obstr{\xi} = \obstr{\xi'} \text{ implies } \pi(\xi) = \pi(\xi').
$$
```
## Reachability probabilities in POMDPs

### Problem statements
```{warning} Assumptions
We assume that target states have a unique and exclusive target observation $z_\mathrm{target}$,
i.e., for $s \in T$, $\obsfun(s) = z_\mathrm{target}$ and for $s\not\in T$, $\obsfun(s) \neq z_\mathrm{target}$.
```

```{admonition} Problem: Standard verification problem for reachability probabilities
Given POMDP $\mdp$, a set of target states $T$, a _threshold_ $\lambda \in \mathbb{Q}$ and $\bowtie \in \{\leq,\geq\}$, decide whether

$$
\pr^\pi_\mdp(\lozenge T) \bowtie \lambda
$$

holds for **all** observation-based policies.
```

```{admonition} Problem: Standard planning problem for reachability probabilities
Given POMDP $\mdp$, a set of target states $T$, $\lambda \in \mathbb{Q}$ and $\bowtie$, compute **some** observation-based policy $\pi$ such that

$$
\pr^\pi_\mdp(\lozenge T) \bowtie \lambda .
$$
```


### Difficulty of the problem
Before we go into details, let us review the famous cheese-maze POMDP.
Even in this simple POMDP, adding memory is important to act optimally.
````{prf:example} Optimal observation-based policies require memory
:label: ex:pomdp:cheesemaze1
```{code-cell} python
:tags: [hide-input]
from stormvogel.examples import create_cheese_maze
cheese_maze = create_cheese_maze()
plot_model_pydot(cheese_maze)
```
Consider the cheese maze POMDP with three vertical corridors.
The agent starts uniformly in one of the (upper) cells of the three corridors; not on the top row.
All cells within the corridors share the same observation $\mathit{NS}$, so from the current observation alone the agent cannot tell which corridor it is in.
The middle corridor leads to the cheese; the outer two lead to the dragon.
It is possible to reach the cheese with probability one.
An optimal strategy first moves upwards. Once it hits the top row, it has uniquely determined the agent's position.
It then moves to the correct corridor and descends. 
A memoryless policy must apply the same action in every corridor --- with observation $\mathit{NS}$.
Then, the agent must never descend while there is positive probability of being in a dragon corridor.
But a policy that never descends never reaches the cheese either.
Thus, a memoryless policy cannot reach the cheese with probability one. 
````
Sometimes, memoryless policies are sufficient, but randomization may then still be necessary:
````{prf:example} Optimal memoryless policies require randomization
:label: ex:pomdp:cheesemaze2
Consider the cheese maze, and assume the agent starts in one of the two top-corner cells (observations $\mathit{ES}$ or $\mathit{SW}$, which are distinguishable).
Even though the starting position is known, reaching the middle corridor requires traversing $\mathit{EW}$ cells on the top row, which are all indistinguishable.
In a memoryless setting, the action taken in an $\mathit{EW}$ cell must be the same no matter how the agent arrived there.
A deterministic memoryless policy fixes a single direction—left or right—at every $\mathit{EW}$ cell, so from one of the two starting corners the agent always moves away from the middle corridor and can never reach the cheese.
A randomising policy, by contrast, can assign a small probability to descending at each $\mathit{NS}$ cell and a high probability to continuing along the top row.
This way, whenever the agent happens to be above the cheese corridor, there is a positive probability of descending correctly, so the cheese is reached with probability one eventually.
No deterministic memoryless policy achieves this.
````
In general, we cannot give an upper bound on the amount of memory necessary. 

````{prf:example} Supremum not attained
:label: ex:pomdp:supnotattained
Consider a POMDP with states $L_l$, $L_r$, $R_l$, $R_r$, goal state $g$, and failure state $f$.
States $L_l$ and $R_l$ share observation $l$; states $L_r$ and $R_r$ share observation $r$: the agent sees only the last signal, not whether the hidden type is $L$ or $R$.
The start state distributes uniformly over all four, so both hidden types are initially equally likely to be true.
The three actions are $\mathsf{listen}$, $\mathsf{guessL}$, and $\mathsf{guessR}$.
Under $\mathsf{listen}$, $L$-states transition to $L_l$ with probability $\tfrac{9}{10}$ and $L_r$ with probability $\tfrac{1}{10}$; $R$-states transition to $R_r$ with probability $\tfrac{9}{10}$ and $R_l$ with probability $\tfrac{1}{10}$ — so the signal is informative but noisy.
Under $\mathsf{guessL}$, the agent moves to $g$ from any $L$-state and to $f$ from any $R$-state; $\mathsf{guessR}$ is symmetric.
```{code-cell} python
:tags: [hide-input]
from stormvogel.examples import create_sup_not_attained_pomdp
sup_model = create_sup_not_attained_pomdp()
plot_model_pydot(sup_model)
```

**The supremum equals 1.**
For any $n$, the agent can apply $\mathsf{listen}$ $n$ times and then guess the majority signal seen.
By the law of large numbers, the probability of guessing correctly approaches $1$ as $n \to \infty$, so $\sup_\pi \pr^\pi(\lozenge g) = 1$.

**No policy attains it.**
Every finite observation history has positive probability under both hidden states: for any sequence $o_1 \dots o_k$,
$$\pr(o_1\dots o_k \mid L) > 0 \quad \text{and} \quad \pr(o_1\dots o_k \mid R) > 0.$$
Therefore, whenever a policy guesses, there is positive probability of being in the wrong hidden state and reaching $f$.
Thus $\pr^\pi(\lozenge g) < 1$ for every policy $\pi$, and the supremum is not attained.
````
This example is key to proving the following general theorem.
```{prf:theorem} Optimal policies do not need to exist
:label: thm:pomdp:nooptimal
For a POMDP $\mdp$ and target $T$, the supremum
$$\sup_{\pi\ \text{observation-based}} \pr^\pi_\mdp(\lozenge T)$$
need not be attained: there may be no observation-based policy $\pi^*$ achieving this value.
```
Indeed, the unbounded amount of memory is at the root of an undecidability statement.

```{prf:theorem} Undecidable
:label: thm:pomdp:undecidable
The standard verification and planning problems for POMDPs with reachability objectives are undecidable.
```

## Beliefs 
Above, we have seen that optimal policies must rely on information from the history. 
However, it turns out that the precise history is not necessary for optimal decisions. 
Instead, the policy depends on the belief, i.e., a distribution that captures the likelihood of being in a particular state,
given the history of observations seen and actions taken. 

````{prf:example} 4-State POMDP — Running Example
:label: ex:pomdp:4state
We use the following POMDP as a running example throughout this chapter.
States $s_1$ and $s_2$ share observation $z$ and are therefore indistinguishable to the agent.
Three actions are available: $a$ leads deterministically to the target from $s_1$ and to the sink from $s_2$; $b$ mixes the hidden state; $c$ gives a partial chance of reaching the target from either state.
```{code-cell} python
:tags: [hide-input]
from stormvogel.examples import create_4state_reachability_pomdp_variantb
model = create_4state_reachability_pomdp_variantb()
obsstates = sorted(model.compute_states_per_observation()[model.get_observation("z")], key=lambda s: s.friendly_name)
s1 = obsstates[0]
s2 = obsstates[1]
plot_model_pydot(model)
```
````

### Belief updates

```{prf:definition} Belief update
:label: def:pomdp:beliefupdate
In a belief $b$, taking action $a$ yields an updated belief as follows. Let
$$
P(b,a,z') := \sum_{s \in S_{\obsfun(b)}} b(s) \cdot \sum_{s' \in S_{z'}} \delta(s,a,s')
$$
denote the probability of observing $z' \in \Obs$ upon taking action $a \in A$ in belief $b$. 

If $P(b,a,z') > 0$, the corresponding successor belief $b' = \llbracket b \mid a, z' \rrbracket$ with $\obsfun(b') = z'$ is defined component-wise as
$$
\llbracket b \mid a, z' \rrbracket(s') :=
\frac{\sum_{s \in S_{\obsfun(b)}} b(s) \cdot \delta(s,a,s')}
{P(b,a,z')}
$$
for all $s' \in S_{z'}$. Otherwise, $\llbracket b \mid a, z' \rrbracket$ is undefined.
```
```{prf:definition} Belief for a path
:label: def:pomdp:belief
Given a $\xi = s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} \dots$, the _belief_ $b_\xi \in \Distr{S}$ is defined inductively:
- For a path of length zero $\xi = s_0$: $b_\xi(s) = \indicator{s = s_0}$.
- For an extended path: $$b_{\xi \cdot a \cdot s'} = \llbracket b_\xi \mid a, \obsfun(s') \rrbracket$$.
```
Naturally, the belief for two paths with the same observation trace coincides.

````{prf:example} Belief updates on the 4-state POMDP
We compute belief updates on the running example (@ex:pomdp:4state), starting from the uniform belief $\binit = \{s_1 \mapsto \tfrac{1}{2}, s_2 \mapsto \tfrac{1}{2}\}$.
```{code-cell} python
:tags: [hide-input]
from stormvogel.teaching.belief import initial_belief, belief_update, belief_trace, belief_table

trace = [("b", "z"), ("b", "z")]
belief_table(belief_trace(model, initial_belief(model, "z"), trace), trace)
```
In the supremum-not-attained POMDP (@ex:pomdp:supnotattained), repeated applications of $\mathsf{listen}$ concentrate the belief towards the true hidden state.
```{code-cell} python
:tags: [hide-input]
trace = [("listen", "l")] * 4
belief_table(belief_trace(sup_model, initial_belief(sup_model, "l"), trace), trace)
```
The belief weight on the $L$ variants grows with each $l$ observation, but never reaches $1$ — reflecting that the hidden state can never be determined with certainty.
````
### Belief-based policies
Given this notion of belief and the mapping from an observation trace to a belief, 
we now introduce policies that make decisions solely based on the belief of a path/of a trace.
```{prf:definition} Belief-based policies
:label: def:pomdp:beliefpolicies
A policy $\pi$ is _belief-based_ if there exists $\hat\pi \colon \Distr{S} \rightarrow \Distr{A}$ such that
$$\pi(\xi) = \hat\pi(b_\xi) \quad \text{for all } \xi \in \Paths.$$
```
```{prf:lemma} Belief-based implies observation-based
:label: lem:pomdp:beliefobs
Every belief-based policy is observation-based. 
```
By induction on the definition of $b_\xi$, equal observation traces yield equal beliefs: $\obstr{\xi} = \obstr{\xi'}$ implies $b_\xi = b_{\xi'}$. 
Thus any belief-based policy is observation-based.

The converse does not hold in general: an observation-based policy may use the full observation history in a way that distinguishes two paths with equal beliefs, e.g., by counting steps or recalling past actions.

```{prf:theorem} Belief-based policies are sufficient (for reachability)
:label: thm:pomdp:beliefsufficient
For any POMDP $\mdp$,
$$\sup_{\pi\ \text{observation-based}} \pr^\pi_\mdp(\lozenge T) = \sup_{\pi\ \text{belief-based}} \pr^\pi_\mdp(\lozenge T).$$
```

### The belief MDP
Since belief-based policies are sufficient and the belief captures everything about the history that is relevant for future decisions, we can replace hidden states entirely with belief distributions, reducing the POMDP to a standard (infinite-state) MDP.
```{prf:definition} Belief MDP
:label: def:pomdp:beliefmdp
The _belief MDP_ of POMDP $\mathcal{M}$ is the (infinite-state) MDP
$$
\mathcal{M}^B = (\mathcal{B}, A, \delta^B),
$$
with transition function
$$
\delta^B(b,a,b')
= [\, b' = \llbracket b \mid a,z' \rrbracket \,]
   \cdot P(b,a,z')
$$
where $z' = \obsfun(b')$.

For a given initial state $\sinit$, the initial belief is
$$
\binit = \{ \sinit \mapsto 1 \}
$$
```
This belief MDP is naively continuous state, 
but for finite POMDPs, the reachable state space of a belief MDP is infinite but countable.
```{prf:lemma} Belief MDP size and shape
:label: lem:pomdp:beliefmdpsize
- The (reachable fragment) of a belief MDP has a countable number of states.
- If the POMDP is acyclic, the belief MDP is acyclic and finite.
```
The optimal reachability probability in a belief MDP coincides with the optimal reachability probability in the POMDP.
```{prf:theorem}
:label: thm:pomdp:beliefmdpequal
For POMDP $\mdp$ with target $T \subseteq S$, let $T^B = \{ b \in \mathcal{B} \mid \mathrm{supp}(b) \subseteq T \}$. Then
$$\sup_{\pi\ \text{observation-based}} \pr^\pi_\mdp(\lozenge T) = \sup_{\pi \in \MdPolicies} \pr^\pi_{\mathcal{M}^B}(\lozenge T^B).$$
```
In particular, it suffices to consider deterministic belief-based policies!

### Belief-value functions
For the belief MDP, we can consider the standard value function that maps a state of that MDP to a value of interest, in this case the maximal reachability probability.
This function thus maps POMDP beliefs to probabilities,  $V\colon \mathcal{B} \rightarrow [0,1]$.
This function is of interest independent of the belief MDP.

Two structural properties are particularly relevant.
First, under any fixed observation-based policy the belief-value function is affine in the belief.
```{prf:lemma}
:label: lem:pomdp:affine
For any observation-based policy $\pi$, $V^\pi(b) = \alpha^\pi \cdot b$ for some vector $\alpha^\pi \in [0,1]^S$.
```
For a fixed observation-based policy $\pi$, the action taken at each step is determined by the observation-action history alone.
The probability of any state trajectory $\xi = s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} \cdots$ under $\pi$ is therefore $b(s_0) \cdot \prod_t \delta(s_t, a_t, s_{t+1})$, which is linear in $b(s_0)$.
Summing over all target-reaching trajectories gives
$$V^\pi(b) = \sum_{s \in S} b(s) \cdot \underbrace{\pr^\pi(\lozenge T \mid s_0 = s)}_{\alpha^\pi(s)} = \alpha^\pi \cdot b.$$

````{prf:example} Alpha vectors on the 4-state POMDP
:label: ex:pomdp:alphavectors
Consider the two single-step memoryless policies on @ex:pomdp:4state: always take action $a$, or always take action $c$.
For policy $\pi_a$: from $s_1$ action $a$ reaches the target with probability $1$; from $s_2$ with probability $0$.
For policy $\pi_c$: from $s_1$ action $c$ reaches the target with probability $\tfrac{1}{4}$; from $s_2$ with probability $\tfrac{4}{5}$.
The two alpha vectors are therefore $\alpha^{\pi_a} = (1,\, 0)$ and $\alpha^{\pi_c} = (\tfrac{1}{4},\, \tfrac{4}{5})$, and the value functions
$$V^{\pi_a}(b) = b(s_1), \qquad V^{\pi_c}(b) = \tfrac{1}{4}\,b(s_1) + \tfrac{4}{5}\,b(s_2)$$
are both linear in $b$.
```{code-cell} python
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

b1_range = np.linspace(0, 1, 200)
v_a = b1_range
v_c = (1/4) * b1_range + (4/5) * (1 - b1_range)

plt.plot(b1_range, v_a, label=r"$V^{\pi_a}(b)$")
plt.plot(b1_range, v_c, label=r"$V^{\pi_c}(b)$")
plt.xlabel(r"$b(s_1)$")
plt.ylabel("Reachability probability")
plt.legend()
plt.tight_layout()
plt.show()
```
````

Such an affine function is called an _alpha vector_:
```{prf:definition} Alpha vector
:label: def:pomdp:alpha
An _alpha vector_ is a function $\alpha \colon S \rightarrow [0,1]$.
For a belief $b \in \mathcal{B}$, the value under $\alpha$ is the dot product, i.e.,
$$\alpha \cdot b = \sum_{s \in S} \alpha(s) \cdot b(s).$$
```
```{prf:lemma}
The belief-value function for a fixed observation-based policy $\pi$ is an alpha-vector.
```

Second, the optimal belief-value function is the pointwise supremum over alpha vectors:
$$V^*(b) = \sup_{\alpha \in \Gamma} \; \alpha \cdot b,$$
where $\Gamma$ is the set of alpha vectors representing all relevant (in general, belief-based deterministic) policies.
```{prf:lemma}
:label: lem:pomdp:convex
$V^*(b)$ is convex.
```
Note that if it suffices to only consider a finite set of such policies, $V^*(b)$ is piecewise linear.


# Algorithms for POMDPs
Since exact computation is undecidable (@thm:pomdp:undecidable) and an optimal policy need not exist (@thm:pomdp:nooptimal), in practice one computes bounds on the optimal value $V^*(\binit)$.
We construct a number of different algorithms,
for either computing lower bounds (mostly for planning) or upper bounds (mostly for verification) of the maximal reachability probability.
The ideas can be lifted to minimal reachability probabilities and to expected reachability rewards and discounted total rewards as usual. 
Supporting richer temporal properties is a bit more involved.

We split the algorithmic ideas into 
1. operating directly on a finite MDP abstraction, 
2. operating over weaker policy classes,
3. operating over stronger policy classes, and
4. operating symbolically in belief space. 

## Algorithms based on finite MDPs
The main idea for these algorithms is simple: We construct a finite MDP that relates to the POMDP and use standard techniques.
In particular, we typically construct a finite MDP that abstracts the infinite belief MDP.
###  Finite belief MDP unfolding
From a verification perspective, operating immediately on the belief MDP is very convenient. 
As this MDP is infinite in general, we may explore a finite fragment of this MDP.
The _finite belief MDP unfolding_ explores $\mathcal{M}^B$ from the initial belief $\binit$, expanding at most $K$ distinct belief states.
Any successor belief that falls outside the budget is called a _frontier belief_.
To make the finite fragment a proper MDP, we can add transitions from the frontier belief states to either a target or sink state.
The way we handle these frontier beliefs is generally controlled by a cutoff function $c\colon S \rightarrow [0,1]$.

```{prf:definition} Finite belief MDP
Fix a POMDP $\mathcal{M}$, a budget $K \in \mathbb{N}$, and a cutoff function $c\colon S \to [0,1]$.
Let $\mathcal{B}^K \subseteq \mathcal{B}$ be the set of at most $K$ distinct beliefs discovered by BFS from $\binit$.
The _finite belief MDP_ $\mathcal{M}^{B,K,c}$ agrees with $\mathcal{M}^B$ on all beliefs in $\mathcal{B}^K$, and associates with each frontier belief $b_f \notin \mathcal{B}^K$ a single absorbing "cut" action:
$$
\delta^{B,K,c}(b_f, \mathrm{cut}, T^B) = c \cdot b_f, \qquad \delta^{B,K,c}(b_f, \mathrm{cut}, \bot) = 1 - c \cdot b_f,
$$
where $c \cdot b_f = \sum_{s \in S} c(s)\, b_f(s)$ and $\bot$ is a fresh absorbing sink state.
```

```{prf:theorem} Over- and under-approximation
Let $V^*$ be the optimal reachability probability in $\mathcal{M}^B$, and $V^{K,c}$ the optimal reachability probability in $\mathcal{M}^{B,K,c}$.
- If $c(s) \geq V^*(s)$ for all $s \in S$: $V^{K,c}(\binit) \geq V^*(\binit)$ (over-approximation).
- If for some policy $\pi$, $c(s) \leq V^\pi(s)$ for all $s \in S$: $V^{K,c}(\binit) \leq V^*(\binit)$ (under-approximation).

In particular, $c \equiv 1$ is always optimistic and $c \equiv 0$ is always pessimistic.
As $K \to \infty$, both bounds converge to $V^*(\binit)$.
```
We note the asymetry: Upper bounds are correct using the convexity, 
whereas the lower bounds must consistently under-approximate a single policy (to then also be affine and thus convex).

Note that computing such cut-offs is simple: 
- For lower bounds, picking $\pi$ and evaluating any (e.g., memoryless) observation-based policy yields a pessimistic bound. 
- For upper bounds, assuming the POMDP is fully observable yields an optimistic bound.

However, the computational challenge is to find good cut-off policies with limited overhead. 
We discuss some strategies to do so @sec:fastlowerbounds and @sec:fastupperbounds, respectively.

````{prf:example}
We construct the finite belief MDP for the 4-state POMDP with initial belief $\binit = \{s_1 \mapsto \tfrac{1}{2},\, s_2 \mapsto \tfrac{1}{2}\}$.
```{code-cell} python
:tags: [hide-input]
from fractions import Fraction
from stormvogel.teaching.belief_mdp import belief_mdp

initial_belief = {s1: Fraction(1, 2), s2: Fraction(1, 2)}
bmdp_lower = belief_mdp(model, initial_belief, cutoff=0, max_states=5)
bmdp_upper = belief_mdp(model, initial_belief, cutoff=1, max_states=5)
plot_model_pydot(bmdp_lower)
plot_model_pydot(bmdp_upper)

```
With a restricted budget, frontier beliefs appear.
The pessimistic ($c = 0$) and optimistic ($c = 1$) unfoldings give lower and upper bounds on $V^*$:
```{code-cell} python
:tags: [hide-input]
result_lower = sv.model_checking(bmdp_lower, 'Pmax=? [F "target"]')
result_upper = sv.model_checking(bmdp_upper, 'Pmax=? [F "target"]')
print(f"Lower bound: {result_lower.at_init()}")
print(f"Upper bound: {result_upper.at_init()}")
```
````
The finite exploration above is general, 
but practically only works well if the (reachable fragment of the) belief MDP is small or in presence of good cut-off values.


### Lovejoy grid discretisation
In a verification context, proving that no policy achieves a target with a probability above a threshold is arguably the most important task.
We note that finite belief MDP unrollings tend to converge slowly to useful upper bounds.
On typical problem is that every step in a belief MDP may only change the belief a tiny bit, so belief MDPs quickly grow out of reasonable bounds.
The following complementary approach constructs a finite MDP that yields overapproximations without relying on cut-off values.
The key idea is to ensure finite exploration by fixing a finite grid (a point set) $\hat{\mathcal{B}} \subset \mathcal{B}$ upfront.
Any belief $b' \notin \hat{\mathcal{B}}$ that arises as a successor is not added directly to the state space; 
instead the belief is represented as a convex combination of the nearest grid points.

The value of such a belief that is not in the point set can be approximated from above by using the convexity of the value function.
```{prf:definition} Convex grid interpolation
Let $\hat{\mathcal{B}} = \{\hat{b}_1, \dots, \hat{b}_m\}$ be a finite set of beliefs whose convex hull covers $\mathcal{B}$.
For any belief $b' \in \mathcal{B}$, write $b' = \sum_{i=1}^m \lambda_i \hat{b}_i$ with $\lambda_i \geq 0$, $\sum_i \lambda_i = 1$.
The _grid-interpolated value_ is
$$
\hat{V}(b') = \sum_{i=1}^m \lambda_i \hat{V}(\hat{b}_i).
$$
```
Because $V^*$ is convex, the interpolated value is an over-approximation:
$$
V^*(b') = V^*\!\left(\sum_i \lambda_i \hat{b}_i\right) \leq \sum_i \lambda_i V^*(\hat{b}_i) \leq \hat{V}(b').
$$

This value-level observation motivates a concrete finite MDP construction: rather than storing off-grid beliefs as states, 
redirect every transition that would land on $b' \notin \hat{\mathcal{B}}$ to a distribution over grid beliefs, using the same convex weights $\lambda_i$.

```{prf:definition} Lovejoy grid MDP
Let $\hat{\mathcal{B}}$ be a finite grid whose convex hull covers $\mathcal{B}$.
For each $b' \in \mathcal{B}$, fix convex weights $\lambda(b', \hat{b}') \geq 0$ with $\sum_{\hat{b}' \in \hat{\mathcal{B}}} \lambda(b', \hat{b}') = 1$ such that $b' = \sum_{\hat{b}' \in \hat{\mathcal{B}}} \lambda(b', \hat{b}')\, \hat{b}'$, where $\lambda(\hat{b}', \hat{b}') = 1$ for on-grid beliefs.
The _Lovejoy grid MDP_ $\hat{\mdp}$ has state space $\hat{\mathcal{B}}$ and transition function
$$
\hat{\delta}(\hat{b}, a, \hat{b}')
= \sum_{b' \in \mathcal{B}} \delta^B(\hat{b}, a, b') \cdot \lambda(b', \hat{b}').
$$
```
```{prf:theorem}
The maximal reachability probability in $\hat{\mdp}$ is an upper bound on $V^*(\binit)$.
```
We remark that building the convex weights is not always trivial. 
However, when there are only two states per observation, computing these weights becomes reasonably straightforward using linear interpolation.
````{prf:example}
We apply the Lovejoy construction to the 4-state POMDP (@ex:pomdp:4state) with initial belief $\binit = \{s_1 \mapsto \tfrac{1}{2}, s_2 \mapsto \tfrac{1}{2}\}$ and grid resolution $k = 8$.
```{code-cell} python
:tags: [hide-input]
from stormvogel.teaching.lovejoy import lovejoy_grid_mdp

lovejoy_mdp = lovejoy_grid_mdp(model, {s1: Fraction(1, 2), s2: Fraction(1, 2)}, k=8)
plot_model_pydot(lovejoy_mdp)
result = sv.model_checking(lovejoy_mdp, 'Pmax=? [F "target"]')
print(f"Lovejoy upper bound: {result.at_init()}")
```
Increasing $k$ refines the grid and tightens the upper bound.
````

````{prf:example}
We apply the Lovejoy construction to the supremum-not-attained POMDP (@ex:pomdp:supnotattained).
Note that the upper bound here is trivially one (as it is also the supremum). 
However, the example nicely shows the approximation.
We take the initial belief $\binit = \{L_l \mapsto \tfrac{1}{2}, R_l \mapsto \tfrac{1}{2}\}$ and grid resolution $k = 8$.
```{code-cell} python
:tags: [hide-input]
l_states = sup_model.compute_states_per_observation()[sup_model.get_observation("l")]
L_l_state = next(s for s in l_states if s.friendly_name == "L_l")
R_l_state = next(s for s in l_states if s.friendly_name == "R_l")

lovejoy_sup_mdp = lovejoy_grid_mdp(sup_model, {L_l_state: Fraction(1, 2), R_l_state: Fraction(1, 2)}, k=8)
plot_model_pydot(lovejoy_sup_mdp)
result_sup = sv.model_checking(lovejoy_sup_mdp, 'Pmax=? [F "target"]')
print(f"Lovejoy upper bound: {result_sup.at_init()}")
```
````
(sec:fastlowerbounds)=
## Lower bounds from memoryless policies
The optimal policy for a POMDP may in general require unbounded memory.
A natural restriction is to _memoryless_  policies, which choose an action based only on the current observation.
We have seen that for some POMDPs, memoryless policies are actually sufficient.

```{prf:definition} Memoryless observation-based policy
An observation-based memoryless policy $\pi$ is a function $\pi\colon \Obs \to \Distr{A}$.
```
Since every memoryless policy is a special case of an observation-based policy:
$$\sup_{\sigma\ \text{memoryless}} \pr^\sigma_\mdp(\lozenge T) \;\leq\; V^*(\binit).$$

Optimising over memoryless policies therefore yields a lower bound on the optimal value. 
Memoryless policies also often yield helpful bounds for finite belief MDP constructions.

### Reduction to a parametric Markov chain

A memoryless policy is fully determined by action probabilities $y_{z,a} \in [0,1]$ for each observation $z \in \Obs$ and action $a \in \EnAct{z}$, subject to $\sum_{a} y_{z,a} = 1$.
Because the policy is observation-based, all states $s \in S_z$ sharing observation $z$ must use the same distribution, so they share the same parameters.
This leads naturally to a [parametric Markov chain (pMC)](parametric.md), extending the [corresponding pMC for MDPs](parametric.md#sec:pmc:representingpolicies) to the POMDP setting by grouping parameters by observation class.

```{prf:definition} Corresponding pMC for POMDP
Extending the [corresponding pMC for MDPs](parametric.md#sec:pmc:representingpolicies), the _corresponding pMC_ $D_\mdp$ for POMDP $\mdp$ has the same state space and introduces parameters $\mathbf{y} = \{y_{z,a} \mid z \in \Obs,\, a \in \EnAct{z}\}$, shared across all states with the same observation.
The transition function is
$$
\delta_{D}(s, s') =
\begin{cases}
\delta(s, a, s') & \text{if } \EnAct{\obsfun(s)} = \{a\}, \\[4pt]
\displaystyle\sum_{a \in \EnAct{\obsfun(s)}} y_{\obsfun(s),\,a} \cdot \delta(s, a, s') & \text{otherwise.}
\end{cases}
$$
```

````{prf:example}
Applying the transformation to the 4-state POMDP: states $s_1$ and $s_2$ share observation $z$, so they get the same policy parameters.
```{code-cell} python
:tags: [hide-input]
from stormvogel.teaching.policy_to_pmc import policy_to_pmc

pmc = policy_to_pmc(model)
plot_model_pydot(pmc)
```
````

Every memoryless policy $\pi$ corresponds to a well-defined valuation $\val$ of $\mathbf{y}$, and vice versa: the induced Markov chain $\mdp[\pi]$ equals the instantiated pMC $D_\mdp[\val]$.
The reachability probability $f(\val) = \pr^\sigma_\mdp(\lozenge T)$ is therefore a rational function of $\mathbf{y}$, as studied in the parametric chapter.

```{prf:theorem}
$$
\sup_{\sigma\ \text{memoryless}} \pr^\sigma_\mdp(\lozenge T)
\;=\;
\max_{\val\ \text{well-defined for}\ D_\mdp} f(\val).
$$
```

This reduces finding the best memoryless policy to **parameter synthesis** for pMCs: maximising the rational function $f$ over the policy simplex.
The number of parameters is $O(|\Obs| \cdot \max_z |\EnAct{z}|)$.

### Adding memory
Similar to goal unfoldings, specific memorystructures can be folded into the POMDP. 
Asking for a memoryless policy on this larger POMDP is then equivalent to asking 
for a finite state controller in a smaller POMDP. 
While the POMDP grows only polynomially in the number of memory states added, a polynomial growth in the number of parameters makes this quickly infeasible.


### Hardness of the problem
The existence of a memoryless policy satisfying a reachability threshold is indeed ETR-complete, via the polynomial-time equivalence between this problem and feasibility synthesis for pMCs.
Note that these notes do not show the direction from parametric models to POMDPs.

(sec:fastupperbounds)=
## Upper bounds from revealing information

A natural upper bound on the belief value function $V^*(b)$ is obtained by giving the policies more information, i.e., 
by optimising over a larger set of policies. Optimising over larger sets of policies can be simpler: 
The simplest larger class of policies to consider are MDP policies that fully observe the current state. 
The optimal value in for those policies (i.e., the optimal value in the underlying MDP)
is a natural upper bound on the optimal value under observation-based policies. 
We consider this upper bound and a tighter variation thereof.

### MDP upper bound
Lifting the fully observable MDP value function $V_\mathrm{MDP}$ to beliefs by expectation gives
$$
V_\mathrm{MDP}(b) = \sum_{s \in S} b(s)\, V_\mathrm{MDP}(s).
$$
This is optimistic: different states in the belief support can implicitly choose different optimal actions, as if the agent knew the hidden state exactly.

### QMDP

The _QMDP approximation_ requires a single action to be chosen under the current belief, but then assumes the state becomes fully observable for all future decisions:

$$
V_\mathrm{QMDP}(b)
= \max_{a \in A}
\sum_{s \in S} b(s)
\sum_{s' \in S} \delta(s,a,s')\, V_\mathrm{MDP}(s').
$$

```{prf:theorem}
For every belief $b \in \mathcal{B}$,
$$V^*(b) \;\leq\; V_\mathrm{QMDP}(b) \;\leq\; V_\mathrm{MDP}(b).$$
```

QMDP is therefore always a tighter upper bound than the plain MDP value.

````{prf:example}
```{code-cell} python
:tags: [hide-input]
from stormvogel.examples import create_two_state_commitment_pomdp
from stormvogel.teaching.pomdp_backup import (
    AlphaVI, initial_alpha, make_operator_pomdp_maxreachprob_exact,
    mdp_bound_alpha, qmdp_alphas, plot_value_function_comparison,
)

model = create_two_state_commitment_pomdp()
plot_model_pydot(model)
obsstates = sorted(model.compute_states_per_observation()[model.get_observation("z")], key=lambda s: s.friendly_name)
s1, s2 = obsstates[0], obsstates[1]

op = make_operator_pomdp_maxreachprob_exact(model, "target")
vi = AlphaVI(op, [initial_alpha(model, "target")])
for _ in range(5):
    vi.step()

ax = plot_value_function_comparison(
    [
        (vi.current_alphas,               "exact VI (k=5)"),
        (qmdp_alphas(model, "target"),    "QMDP"),
        ([mdp_bound_alpha(model, "target")], "MDP bound"),
    ],
    s1, s2,
)
```
````

### Fast Informed Bounds and Tighter Informed Bounds
The idea of $V_\mathrm{QMDP}$ can be taken further.
QMDP assumes full observability after a single action; one can instead delay that assumption for $k \geq 1$ steps.
The agent optimises over the next $k$ actions under partial observability — taking into account the observations received — and then falls back to the fully-observable MDP value.
Increasing $k$ yields tighter upper bounds at the cost of an optimisation that is exponential in $k$.
Fast informed bounds (FIB) @Hauskrecht00 and tighter informed bounds (TIB) @KraleKJS025 are two concrete instantiations of this idea that differ in how the $k$-step optimisation is carried out.


## Value iteration in belief space

The Bellman operator lifts naturally to beliefs:
```{prf:definition} Belief-space Bellman operator
:label: def:pomdp:bellmanop
The _belief-space Bellman operator_ $\Phi$ acts on $V \colon \mathcal{B} \rightarrow [0,1]$ by
$$(\Phi V)(b) = \begin{cases} 1 & \text{if } b \in T^B, \\ \displaystyle\max_{a \in A} \sum_{z' \in \Obs} P(b,a,z') \cdot V\!\left(\llbracket b \mid a, z' \rrbracket\right) & \text{otherwise.}\end{cases}$$
```
Initialised with the target indicator $V_0(b) = \indicator{\mathrm{supp}(b) \subseteq T}$,
one application yields the one-step reachability probability, and iterating converges to $V^*$.

````{prf:example}
Consider @ex:pomdp:4state.
```{code-cell} python
:tags: [hide-input]
from stormvogel.teaching.pomdp_backup import (
      AlphaVI, initial_alpha, make_operator_pomdp_maxreachprob_exact,                                                                                                        
      plot_alpha_vector_iterations,                                                                                                                                    
  )                                                                                                                                                                             
op = make_operator_pomdp_maxreachprob_exact(model, "target")                                                                                                                                                                       
vi = AlphaVI(op, [initial_alpha(model, "target")])                                                                                                                   
iters = [vi.step() for _ in range(4)]                                                                                                                              
                                                                                                                                                                       
fig, axes = plot_alpha_vector_iterations(iters, s1, s2) 
```
````
```{attention}
Content here is missing.
```