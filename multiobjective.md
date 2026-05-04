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

# Multiobjective Model Checking and other Tradeoffs


```{code-cell} python
:tags: [remove-input]
import stormpy
import sympy
import stormvogel as sv
import stormvogel.teaching as teach
import stormvogel.bird as bird
import stormvogel.to_dot

sympy.init_printing()
from IPython.display import Math
import stormvogel.teaching.bellman as bellman
```

In the previous chapter,
we discussed MDP model checking where a reachability probability or a reward had to exceed a threshold or remain below a threshold.
To this end, we considered policies that maximised or minimised the probability or reward. 
In this chapter, we look at richer specifications where we cannot simply optimize one reachability probability or reward.

##  Multiobjective Model Checking
We study MDPs where there are multiple target sets that we aim to reach. 
````{prf:example}
Consider the MDP below. 
If the goal is to only optimize reaching the red states, taking α in $s_0$ is optimal.
If the goal is to reach the blue states, taking β in $s_0$ is optimal.
However, if both goals are equally relevant, then taking γ in $s_0$ may be preferred.
```{code-cell} python
:tags: [remove-input]
# Create MDP
mdp = sv.model.new_mdp()

# Initial state
s_init = mdp.initial_state
s_init.set_friendly_name("s0")

# Actions
act_alpha = mdp.new_action("α")
act_beta  = mdp.new_action("β")
act_gamma = mdp.new_action("γ")

# Terminal states
s_red    = mdp.new_state("red", friendly_name="s1")
s_white1 = mdp.new_state(friendly_name="s2")
s_blue   = mdp.new_state("blue", friendly_name="s3")
s_blue2 =  mdp.new_state("blue", friendly_name="s4")

# Direct probabilistic choices from s_init
s_init.set_choices({
    act_alpha: [(0.9, s_red), (0.1, s_blue)],
    act_beta:  [(0.6, s_blue), (0.4, s_white1)],
    act_gamma:  [(0.5, s_blue2), (0.5, s_white1)]
})
s_blue2.set_choices({
    act_alpha: [(0.7, s_red), (0.3, s_white1)],
    act_beta: [(0.6, s_blue), (0.4, s_white1)]
})

# Add self-loops for terminal states
mdp.add_self_loops()

# Plot
sv.to_dot.plot_model_pydot(mdp, state_colors={"red": "red", "blue": "blue"}, default_fill="white")
```
````
With multiobjective model checking queries, we make the notion of achieving multiple goals concrete.

### What are Multiobjective Model Checking queries?
Throughout this chapter, we fix an MDP $\mdp$ with $n$ target sets $T_1, \dots, T_n$, $T_i \subseteq S$.
The central multiobjective model checking problem is the following:
```{prf:definition} Multiobjective Model Checking Achievability Query
Fix an MDP $\mdp$ with $m$ target sets $T_1, \dots, T_m$, $T_i \subseteq S$. 
A policy $\pi$ _achieves_ a point $\vec{\lambda}$, if $$\bigland_{i \leq m} \pr_\pi(\lozenge T_i) \geq \lambda_i.$$
We call $\pi$ a _witnessing policy_ for $\vec{\lambda}$.
A point $\vec{\lambda} \in \mathbb{R}^m$ is _achievable_ if there exists a witnessing policy, i.e., $\vec{\lambda}$ is achievable if
$$ \exists \pi \in \Policies. \bigland_{i \leq m} \pr_\pi(\lozenge T_i) \geq \lambda_i. $$
We denote the set of achievable points with $\Ach_\mdp(T_1, \dots, T_m)$. 
```
We simply write $\Ach$ whenever $\mdp$ and $T_1, \dots, T_m$ is clear from the context.

````{prf:example}
Continuing the example above, the policy $\pi$ with $\pi(s_0) = γ$ and $\pi(s_4) = α$ witnesses the achievability of $(0.5, 0.5)$.
While we can reach the red states with probability $0.9$ and the blue states with probability $0.6$, the point $(0.9, 0.6)$ is not achievable.
````

Note that the literature considers a much wider variation of combinations of different objectives,
including the combination of minimal and maximal reachability probabilities,
the extension to rewards, cost-bounded reachability probabilities, LTL formulas, etc., as well as their combinations.

### On witnessing policies

First, deterministic policies are not sufficient:
```{prf:lemma}
There exist MDPs with $2$ targets and achievable points such that every witness is randomizing, i.e., where no deterministic witnessing policy exists.
```
We provide a proof by the following example.
````{prf:example}
Consider the MDP below and the point $\lambda = (0.5, 0.5)$.
```{code-cell} python
:tags: [remove-input]
# Create MDP
mdp_rnd = sv.model.new_mdp()

s0 = mdp_rnd.initial_state
s0.set_friendly_name("s0")
# Add states with friendly names
s1 = mdp_rnd.new_state("A", friendly_name="s1")
s2 = mdp_rnd.new_state("B", friendly_name="s2")
act_a = mdp_rnd.new_action("a")
act_b = mdp_rnd.new_action("b")
s0.set_choices({act_a: [(1,s1)], act_b: [(1,s2)]})
# Add sink self-loops (important for well-formed MDP)
mdp_rnd.add_self_loops()
          
sv.to_dot.plot_model_pydot(mdp_rnd, positions={s0: (0,0), s1: (2,0), s2: (-2,0)}, state_colors={"A": "red", "B": "blue"}, default_fill="white")
```
A policy that picks action $a$ and action $b$ both with probability half achieves exactly $(0.5, 0.5)$.
Thus, $\lambda$ is achievable. 
However, no deterministic policy achieves $\lambda$. Specifically, in the MDP, there are two deterministic policies, that either pick $a$ or $b$ in the initial state. 
The first policy achieves point $(1,0)$, the second policy achieves $(0,1)$. Thus, both do not achieve $\lambda$.

````


Second, memoryless policies are not sufficient:
```{prf:lemma}
There exist MDPs with $2$ targets and achievable points such that every witness requires memory, i.e., where no memoryless witnessing policy exists.
```
````{prf:example}
:label: ex:multiobjective:memorynecessary
Consider the MDP below and the point $\lambda = (1, 1)$.
```{code-cell} python
:tags: [remove-input]
# Create MDP
mdp_mem = sv.model.new_mdp()

s0 = mdp_mem.initial_state
s0.set_friendly_name("s0")
# Add states with friendly names
s1 = mdp_mem.new_state("A", friendly_name="s1")
s2 = mdp_mem.new_state("B", friendly_name="s2")
act_a = mdp_mem.new_action("a")
act_b = mdp_mem.new_action("b")

s0.set_choices({act_a: [(1,s1)], act_b: [(1,s2)]})
# s1 goes back to s0
s1.set_choices({
    act_a: [(1.0, s0)]
})

# Add sink self-loops (important for well-formed MDP)
mdp_mem.add_self_loops()
          
sv.to_dot.plot_model_pydot(mdp_mem, positions={s0: (0,0), s1: (2,0), s2: (-2,0)}, state_colors={"A": "red", "B": "blue"}, default_fill="white")
```
Specifically, a witnessing policy visits first the red state $s_1$ before visiting the blue state $s_2$. 
In particular, the policy changes its mode once we enter $s_1$: The policy now only wants to optimize reaching the blue state $s_2$. 
As we formalise [later](#def:multiobjective:unfolding), the policy must thus track which target sets have already been visited. 
````

However, it suffices to consider randomising finite-memory policies. 
```{prf:definition}
A _randomising finite-memory policy_ is a policy that can be represented as 
an annotated finite state machine, i.e., by a tuple $\langle Q, \alpha, \delta, q_0 \rangle$
 with 
 - a finite set of nodes $Q$ and initial node $q_0$,
 - an annotation $\alpha\colon Q \times S \rightarrow \Distr{A}$,
 - transition relation $Q \times S \rightarrow Q$.
 
The finite state machine defines policy $\pi$ with $$\pi(s_0\dots s_n) = \alpha(\delta(q_0, s_1\dots s_{n-1}), s_n).$$  
```
```{prf:theorem}
A point $\vec{\lambda}$ is achievable iff there exists a randomising finite-memory policy that achieves $\vec{\lambda}$.  
```

### On achievable points
We can look at the geometric properties of $\Ach$. We use notions such as convex and downward closed sets from @app:geometry.


```{prf:theorem}
The set $\Ach$ is convex and downward-closed.
```
Downward closure follows immediately from the definition of achievability. 
The convex closure is a bit more interesting. 
Intuitively, the idea is that if we pick two achievable points $\vec{p}_1, \vec{p}_2$, we can take two witnessing policies $\pi_1, \pi_2$. 
If we now implement a protocol where we flip a coin with probability $c$, we play like policy $\pi_1$ and with probability $1-c$, we play like $\pi_2$.
One needs some care to correctly define such convex combinations of policies. Consider, e.g., that $\pi_1$ stays forever in an MEC and $\pi_2$ leaves that MEC. 
A naive realization of a convex combination of both policies would always leave the MEC. A thorough treatment of this topic can be found, e.g., in @DBLP:phd/dnb/Quatmann23. 
```{prf:definition}
A point $\vec{p} \in \mathbb{R}^m$ is _Pareto-optimal_ for targets $T_1, \dots, T_m$ iff for all $\vec{q} \in \mathbb{R}^m$:
- $\vec{p} \succ \vec{q}$ implies $\vec{q} \in \mathrm{Ach}$, and
- $\vec{q} \succ \vec{p}$ implies $\vec{q} \notin \mathrm{Ach}$.

The _Pareto curve_  is the set
$$
\mathrm{Pareto}
= \left\{ \vec{p} \in \mathbb{R}^m \mid \vec{p} \text{ is Pareto optimal} \right\}.
$$
```

```{prf:lemma}
It holds: $\mathrm{Pareto} \subseteq \Ach$.
```
Specifically, this means that any Pareto-optimal point has a witnessing policy.  
Note that the lemma above is not true for arbitrary objectives that go beyond maximal reachability probabilities.

```{prf:theorem}
There is a finite set of points $P$, such that $\Ach$ is the smallest convex and downward-closed set containing $P$.
```
Even stronger (and key to proving the theorem above), these finite points are achieved by deterministic and (bounded) finite-memory policies.
However, note that $|P|$ can be superpolynomial in the size of the MDP, even if there are just 2 objectives.


### Computing Achievable Points
We consider two computational problems. The first is a simple decision problem.
```{admonition} Problem: Decision Problem - Achievability
Given an MDP $\mdp$ with $m$ target sets $T_1, \dots, T_m$, $T_i \subseteq S$, and a point $\vec{\lambda} \in \mathbb{R}^m$.
Decide whether $\vec{\lambda}$ is achievable. 
```
Beyond deciding achievability of individual points, we can aim to compute the Pareto curve (or rather, the achievable points). 
As the representation of the achievable points can be prohibitively large, we are mostly interested in computing a tight approximation. 
```{admonition} Problem: Achievable Point Approximation
Given an MDP $\mdp$ with $m$ target sets $T_1, \dots, T_m$, $T_i \subseteq S$.
Compute $L, U \subseteq \mathbb{R}^m$ such that:
- $L \subseteq \Ach_\mdp(T_1, \dots, T_m) \subseteq U$, and 
- $\|U - L\| \leq \varepsilon$, for any given $\varepsilon$ and suitable measure on sets. 
```
Computing _the_ set of achievable points is a special case where $\varepsilon = 0$.
Beyond these queries, one can also try to find particular points on the Pareto curve. This is often called a constrained MDP query. 


#### Preprocessing: Goal unfolding
In this preprocessing step, 
we reduce reasoning about randomising finite memory policies to reasoning about randomising memoryless policies.
The key insight used here is that the memory is not used arbitrarily, but simply to track which set of target states has been visited previously.
We then move this memory from the policy into the MDP itself.
```{prf:definition} Goal unfolding
:label:def:multiobjective:unfolding
Given an MDP $\mdp = \langle S, A, \delta_\mathsf{orig} \rangle$ with target states $T_1, \dots, T_m$. 
We define $\mathsf{unfolding}(\mdp)$ as an MDP $$ \langle S \times \{0,1\}^m, A, \delta_{\mathsf{unf}} \rangle $$  
with 
$$
\delta_\mathsf{unf}((s,b), a, (s',c)) = 
\begin{cases} \delta_\mathsf{orig}(s,a,s') & \text{if }c=\mathsf{succ}(b, s') \\ 
0 & \text{otherwise.} \end{cases} 
$$
where $\mathsf{succ}(b,s')_i = b_i \lor \indicator{s'\in T_i}$ for any $i$.
``` 
````{prf:example}
:label:ex:multiobjective:unfolding
In the figure below, 
we apply the unfolding construction on the MDP from @ex:multiobjective:memorynecessary. 
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.multiobjective import goal_unfolding
unfolded, unfolded_bitmap = goal_unfolding(mdp_mem, ["A", "B"], return_state_bits=True)
sv.to_dot.plot_model_pydot(unfolded)
```
````
The correctness of the construction is summarized by the following theorem:
```{prf:theorem} 
- The set of achievable points in $\mdp$ and $\mathsf{unfolding}(\mdp)$ coincide.
- Every achievable point for $\mathsf{unfolding}(\mdp)$ has a memoryless witness.
```
```{prf:lemma}
If all target states are absorbing, then every achievable point has a memoryless witness.
```
We remark that the construction is indeed exponential in the number of target sets (only).

#### Deciding achievability via an LP
We assume an unfolded MDP as in @def:multiobjective:unfolding. We furthermore assume the initial state is not in any $T_i$.
We first show that the decision problem, whether a given point is achievable, can be solved with an LP.
The (standard) LPs for probabilistic model checking have variables for the probability to reach the target and constraints to encode the action selection.
That setup makes it hard to define different reachability probabilities under the same policy.
However, we can take a different perspective (in fact, the dual perspective in the linear programming theory) and consider the probability mass flow through the MDP.

Specifically, we define an LP over variables $y_{s,a}$ for every choice $(s,a)$.
The idea is that $y_{s,a}$ is to encode the expected number of times that a policy chooses $(s,a)$.
Clearly, this number must be non-negative, i.e., $$\text{for all } s \in S, a \in \EnAct{s}: y_{s,a} \geq 0.$$
```{prf:remark} Choice of variables.
One can rewrite this to use variables $y_{s,a,s'}$  for every transition $s \xrightarrow{a} s'$, using the equality $y_{s,a,s'} = \delta(s,a,s')y_{s,a}$.
```
Under this intuition, we can define auxiliary expressions for the expected number of times a state is left (outflow) and the number of times a state is entered (inflow).
$$ 
f_{out}(s) = \sum_{a} y_{s,a} + \indicator{s = \sinit}
$$
and 
$$
f_{in}(s') = \sum_{s}\sum_{a} \delta(s,a,s') \cdot y_{s,a}.
$$
Intuitively, we want to express that the times a state is entered must be equal to the times we leave the state, i.e.,
$$
f_{in}(s) = f_{out}(s).
$$
However, the number of times a state is visited under a particular policy may be infinite, if it is part of a maximal end component.
We make the following two key observations (from @DBLP:conf/tacas/ForejtKNPQ11):
- if there still is a target state that can be reached, a (Pareto) optimal policy has no incentive to stay in an end-component.
- if there is no target state to be reached, the behavior of a policy (and what states it will reach) is not relevant to our property.
We can define the relevant states  as follows:
$$ S_\mathsf{rel} = \{ (s,b) \mid \exists c \geq b. (s,c) \text{ reachable from } (s,b). \} $$[^sreldef]
We will therefore formulate the following constraint:

$$ \text{for all} s \in S_\mathsf{rel}, f_{in}(s) = f_{out}(s). $$

The last remaining set of constraints enforces that the policy is actually achievable.

(def:splusandsminus)=
Consider the sets
$S_{-i} = \{ (s,b) \mid b[i]=0  \}$ and $ S_{+i} = \{(s,b) \mid b[i] = 1 \}, $ 
i.e., the sets of state where $T_i$ has not and has already been visited. 

Observe that along every path, one transitions either zero or one time from $S_{-i}$ to $S_{+i}$. 
Thus, the expected number of such transitions coincides with the probability to go from $S_{-i}$ to $S_{+i}$.
As one starts in $S_{-i}$ (the initial state is not part of $T_i$ by assumption), the probability to go from  $S_{-i}$ to $S_{+i}$ is the same as the probability to reach $S_{+i}$,
which is the same as the probability to reach a $T_i$ state. This justifies the following constraint:
$$ \text{for all } i \leq m \sum_{s \in S_{-i}}\sum_{a}\sum_{s' \in S_{+i}} \delta(s,a,s')\cdot y_{s,a} \geq \lambda_i.$$
If we put these constraints together, we get a sound and complete approach to check achievability.
```{prf:theorem}
Given an [goal-unfolded MDP](#def:multiobjective:unfolding) $\mdp$ and a threshold $\vec{\lambda}$. 
The LP
\begin{align*}
	\text{maximize}\quad & 0 & &   \\
	  & y_{s,a} \geq 0 & & \text{ for all }s \in S, a \in \EnAct{s},  \\
	  & f_{in}(s) = f_{out}(s) & & \text{ for all } s \in S_\mathsf{rel},  \\
	  & \sum_{s \in S_{-i}}\sum_{a}\sum_{s' \in S_{+i}} \delta(s,a,s')\cdot y_{s,a} \geq \lambda_i. & & \text{for all } i \leq m
\end{align*}
is feasible iff $\lambda$ is achievable.
```
The easiest way to prove the correctness of the approach is by considering this as the dual to the more intuitive standard LP.
A witnessing policy $\pi$ can be extracted by setting $$\pi(s) = \argmax_{a} y_{s,a}. $$
````{prf:example}
Consider the [unfolded MDP from above](#ex:multiobjective:unfolding). 
The following LP is feasible iff $(0.5, 0.5)$ is achievable.
```{code-cell} python
:tags: [remove-input]
from  stormvogel.teaching.multiobjective import lp_dual_multireachprob
lp_dual_multireachprob(unfolded, unfolded_bitmap, ["A", "B"], threshold=(0.5,0.5))
```
````
As for single-objective MDP model checking, the LP justifies the following statement.
```{prf:lemma}
Deciding achievability can be done in polynomial time.
```
We remark that the LP can be extended with an objective such that a solution to the LP yields a Pareto optimal policy. 

[^sreldef]: This definition is close to, but not the same as $\{ s \mid \bigvee_i s_i \in \Spos(\lozenge T_i) \}$: Consider absorbing target states.


#### Computing Pareto curves via weighted reachability
We now turn our attention to approximating the Pareto curve. 
We obtain our approximation by computing a subset of the Pareto-optimal policies.
Clearly, the convex and downward closure of the points achieved by these policies is an underapproximation.
From the fact that the policies we compute are indeed Pareto-optimal, we also obtain a set of unachievable points.
The complement of these unachievable points are then an overapproximation of the set of achievable points.
As the achievable points can be represented by finitely many vertices, the algorithm we present also terminates for any precision $\epsilon$.

More specifically, the key theorem to the approach is the following:
```{prf:theorem}
Given $\vec{w} \in [0,1]^m$ and $$ \pi \in \arg\max_{\pi'}  \sum w_i \pr_{\pi'}(\lozenge T_i).$$
Then: $$\vec{p} = (\pr_\pi(\lozenge T_1), \dots \pr_\pi(\lozenge T_m))$$ is on the boundary of the achievable points.
For $w_i \in (0,1]^m$, the point $\vec{p}$ is Pareto-optimal.
```
Clearly, $\vec{p}$ is achievable by $\pi$. The argument that it is also Pareto optimal (for almost all $w$) follows from the fact that every point that dominates $p$ yields a higher weighted reachability.
Assume that one of those points would be achievable by policy $\hat{\pi}$: Then $\hat{\pi}$ demonstrates that $\pi$ is not optimal w.r.t. weighted reachability, which is a contradiction to the definition.

With this theorem, the idea is now to iteratively explore different weight vectors. 
Every weight vector sharpens the overapproximation.
By carefully selecting the weight vectors, we can also ensure we eventually find all vertices of the Pareto curve.
````{prf:example}
In the following, we first optimize for the weight vector $(1, 0)$.
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.pareto import ParetoQuery, explore_pareto
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0)], figsize=(2, 2), legend="outside") 
```
Importantly, we find a policy that achieves $(0.9, 0.1)$ and update the achievable region accordingly.
We can also update the unachievable points (i.e., the complement of the upper bound), by ruling out the hyperplane.

Next, we optimize for the weight vector $(0,1)$.
```{code-cell} python
:tags: [remove-input]
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0), (0.0, 1.0)], figsize=(2, 2), legend="outside") 
```
We now see that the convex hull is formed, in particular, points between  two achievable points must also be achievable.

Next, we take a weight vector $(0.5, 0.3)$ orthogonal to the current face between the two Pareto-optimal points:
```{code-cell} python
:tags: [remove-input]
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0), (0.0, 1.0), (0.3, 0.5)], figsize=(2, 2), legend="outside") 
```
This yields one more Pareto optimal point.

By adding two more weight vectors, we can prove that there are no further achievable points:
```{code-cell} python
:tags: [remove-input]
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0), (0.0, 1.0), (0.3, 0.5), (0.1, 0.35), (0.4, 0.55)], figsize=(2, 2), legend="outside") 
```
````

````{prf:remark} On computing weighted reachability
The weighted reachability query can be expressed as a total (undiscounted) reward query on the unfolding.
Specifically, you get reward $w_i$ for every transition between an $S_{-i}$ and $S_{+i}$, using the [earlier definitions](#def:splusandsminus). 
While the total reward in general is infinite, it is not if the reward collectible in every MEC is zero.
As the reward, by construction, is only awarded upon the first entry of some target set, we can only finitely often pick up reward.
Therefore, the total reward is finite and computable via all standard approaches, see @DBLP:phd/dnb/Quatmann23[Section 4.1.3].
````

#### Variant 3: Convex Hull Value Iteration
```{attention}
Currently skipped.
```

## Tractable hyperproperties
```{attention}
Work in progress
```