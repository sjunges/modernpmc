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

(chap:multiobjective)=
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
Continuing the example above, the policy $\pi$ with $\pi(s_0) = γ$ and $\pi(s_4) = α$ witnesses the achievability of $(0.5, 0.35)$.
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
          
sv.to_dot.plot_model_pydot(mdp_rnd, positions={s0: (0,0), s1: (2,0), s2: (-2,0)}, state_colors={"A": "red", "B": "blue"}, default_fill="white", self_loop_position="n")
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
          
sv.to_dot.plot_model_pydot(mdp_mem, positions={s0: (0,0), s1: (2,0), s2: (-2,0)}, state_colors={"A": "red", "B": "blue"}, default_fill="white",self_loop_position="n")
```
Specifically, a witnessing policy visits first the red state $s_1$ before visiting the blue state $s_2$. 
In particular, the policy changes its mode once we enter $s_1$: The policy now only wants to optimize reaching the blue state $s_2$. 
As we formalise [later](#def:multiobjective:unfolding), the policy must thus track which target sets have already been visited. 
````
Memory is necessary to distinguish which target states have already been visited along a path. 
````{prf:example}
The following with targets $A = \{s_1, s_A\}$ and $B = \{s_1', s_B\}$:
```{code-cell} python
:tags: [remove-input]
mdp_mem2 = sv.model.new_mdp()
s0_m2 = mdp_mem2.initial_state
s0_m2.set_friendly_name("s0")
s1_m2  = mdp_mem2.new_state("A", friendly_name="s1")
s1p_m2 = mdp_mem2.new_state("B", friendly_name="s1'")
s2_m2  = mdp_mem2.new_state(friendly_name="s2")
sA_m2  = mdp_mem2.new_state("A", friendly_name="sA")
sB_m2  = mdp_mem2.new_state("B", friendly_name="sB")
act_p2    = mdp_mem2.new_action("p")
act_al_m2 = mdp_mem2.new_action("α")
act_be_m2 = mdp_mem2.new_action("β")
s0_m2.set_choices({act_p2: [(0.5, s1_m2), (0.5, s1p_m2)]})
s1_m2.set_choices( {act_p2: [(1.0, s2_m2)]})
s1p_m2.set_choices({act_p2: [(1.0, s2_m2)]})
s2_m2.set_choices({
    act_al_m2: [(0.8, sA_m2), (0.2, sB_m2)],
    act_be_m2: [(0.2, sA_m2), (0.8, sB_m2)],
})
mdp_mem2.add_self_loops()
sv.to_dot.plot_model_pydot(mdp_mem2, state_colors={"A": "red", "B": "blue"}, default_fill="white")
```
There is only one state with a nondeterministic choice, $s_2$, and thus 
there are two memoryless deterministic policies, $\pi_\alpha$ corresponding to picking $\alpha$, and $\pi_\beta$ to picking $\beta$.
We obtain:
$$\pr_{\pi_\alpha}(\lozenge A) = \tfrac{1}{2} \cdot 1 + \tfrac{1}{2} \cdot 0.8 = 0.9, \qquad \Pr(\lozenge B) = \tfrac{1}{2} \cdot 0.2 + \tfrac{1}{2} \cdot 1 = 0.6.$$
and symmetrically 
$\pr_{\pi_\beta}(\lozenge A) = 0.6$ and $\pr_{\pi_\beta}(\lozenge B) = 0.9$.
By randomising we can pick $\alpha$ with probability $x$, we can achieve the points $(0.6 + 0.3x,\; 0.9 - 0.3x)$, lying on the line segment from $(0.6, 0.9)$ to $(0.9, 0.6)$.
Every such point satisfies $\pr_\pi(\lozenge A) + \pr_\pi(\lozenge B) = 1.5$.

Using (goal) memory, we can improve upon this policy, if we take $\beta$ at $s_2$ if $A$ was already visited and $\alpha$ if $B$ was already visited.
We then achieve point $(0.9, 0.9)$ and as $\pr(\lozenge A) + \pr(\lozenge B) = 1.8 > 1.5$ it strictly improves upon the memorylessly achievable points.
````

However, it suffices to consider randomising finite-memory policies, i.e., [finite-state controllers (FSCs)](#def:fsc) from the MDP chapter.
```{prf:theorem}
A point $\vec{\lambda}$ is achievable iff there exists an FSC that achieves $\vec{\lambda}$.  
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

Here $\vec{p} \succ \vec{q}$ (read: $\vec{p}$ _dominates_ $\vec{q}$) means $p_i \geq q_i$ for all $i \leq m$ with at least one strict inequality.

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

In other words, $\Ach$ equals the downward closure of the convex hull of its Pareto-optimal points.
In 2D, the Pareto curve is a concave piecewise-linear upper-right boundary, and every point coordinate-wise below some point on that boundary is achievable.


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
Absorbing targets trap the run once visited, so every non-target decision state is always reached before any target has been seen — the goal-memory bit vector is all zeros at every decision point, making it uninformative.

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
After $k$ iterations, let $\vec{p}^{(1)}, \dots, \vec{p}^{(k)}$ be the Pareto-optimal points found so far.
The underapproximation $L_k$ is the downward closure of the convex hull of these points; since each $\vec{p}^{(j)}$ is achievable, every point in $L_k$ is achievable.
The overapproximation $U_k$ is defined by the supporting half-planes induced by each $\vec{p}^{(j)}$; every point outside $U_k$ is not achievable.
As the achievable points can be represented by finitely many vertices, the algorithm terminates for any precision $\epsilon$, at which point $\|U_k - L_k\| \leq \epsilon$.

More specifically, the key theorem to the approach is the following:
```{prf:theorem}
Given $\vec{w} \in [0,1]^m$ and $$ \pi \in \arg\max_{\pi'}  \sum w_i \pr_{\pi'}(\lozenge T_i).$$
Then: $$\vec{p} = (\pr_\pi(\lozenge T_1), \dots \pr_\pi(\lozenge T_m))$$ is on the boundary of the achievable points.
For $w_i \in (0,1]^m$, the point $\vec{p}$ is Pareto-optimal.
```
Clearly, $\vec{p}$ is achievable by $\pi$. The argument that it is also Pareto optimal (for almost all $w$) follows from the fact that every point that dominates $p$ yields a higher weighted reachability.
Assume that one of those points would be achievable by policy $\hat{\pi}$: Then $\hat{\pi}$ demonstrates that $\pi$ is not optimal w.r.t. weighted reachability, which is a contradiction to the definition.

The theorem also gives a mechanism for the overapproximation.
Since $\pi$ is optimal for $\vec{w}$, any achievable point $\vec{q}$ must satisfy $\vec{w}\cdot\vec{q} \leq \vec{w}\cdot\vec{p}$: if some achievable $\vec{q}$ had $\vec{w}\cdot\vec{q} > \vec{w}\cdot\vec{p}$, a witnessing policy for $\vec{q}$ would outperform $\pi$, contradicting optimality.
Each optimal point $\vec{p}^{(j)}$ found with weight $\vec{w}^{(j)}$ therefore yields a supporting half-plane $H(\vec{w}^{(j)}, \vec{p}^{(j)}) \supseteq \Ach$ (see @app:geometry), and the overapproximation after $k$ iterations is their intersection:
$$U_k = \bigcap_{j=1}^{k} H(\vec{w}^{(j)}, \vec{p}^{(j)}).$$
A point outside $U_k$ is provably not achievable; a point inside $U_k$ but not yet witnessed has unknown status.

With this theorem, the idea is now to iteratively explore different weight vectors. 
The theorem above ensures the soundness. 
Completeness of the algorithm relies on the fact that there are only finitely many policies,
but also requires carefully selecting the weight vectors. In particular, 
while every weight vector different to the previous weight vectors sharpens the overapproximation,
we need to be careful to ensure that we find all vertices in finite time. 
However, if we pick the weight vectors orthogonal to one of the existing faces of the Pareto curve, we only need as many iterations as the number of faces + number of vertices of the exact Pareto curve.[^orthogonal]

[^orthogonal]: In 2D, the face between two Pareto-optimal points $\vec{p}^{(1)}$ and $\vec{p}^{(2)}$ is a line segment with direction $\vec{p}^{(2)} - \vec{p}^{(1)}$. A weight vector orthogonal to this face is $(p^{(1)}_2 - p^{(2)}_2,\; p^{(2)}_1 - p^{(1)}_1)$.
The example below also demonstrates this way of picking weight vectors.

````{prf:example}
In the following, we first optimize for the weight vector $(1, 0)$.
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.pareto import ParetoQuery, explore_pareto
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0)], figsize=(2, 2), legend="outside") 
```
Importantly, we find a policy that achieves $(0.9, 0.1)$ and update the achievable region accordingly.
We can also update the unachievable points (i.e., the complement of the upper bound), by ruling out the hyperplane.
Concretely, since $(0.9, 0.1)$ is optimal for $\vec{w} = (1,0)$, every achievable point $\vec{q}$ must satisfy $q_1 \leq 0.9$.

Next, we optimize for the weight vector $(0,1)$.
```{code-cell} python
:tags: [remove-input]
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0), (0.0, 1.0)], figsize=(2, 2), legend="outside") 
```
Let $\vec{p}^{(2)}$ be the optimal point for weight $(0,1)$.
We now have $U_2 = H((1,0), \vec{p}^{(1)}) \cap H((0,1), \vec{p}^{(2)})$, i.e., every achievable point satisfies $q_1 \leq p^{(1)}_1$ and $q_2 \leq p^{(2)}_2$.
The underapproximation $L_2$ is the downward closure of the segment between $\vec{p}^{(1)}$ and $\vec{p}^{(2)}$.
Points between the two Pareto-optimal points must also be achievable (by convexity), but there may be a gap between $L_2$ and $U_2$.

Next, we take a weight vector $(0.5, 0.9)$ orthogonal to the current face between the two Pareto-optimal points:
```{code-cell} python
:tags: [remove-input]
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0), (0.0, 1.0), (0.5, 0.9)], figsize=(2, 2), legend="outside") 
```
Let $\vec{p}^{(3)}$ be the new optimal point.
The overapproximation tightens to $U_3 = U_2 \cap H((0.5, 0.9), \vec{p}^{(3)})$, and $L_3$ gains $\vec{p}^{(3)}$ as a new vertex.
Note that we have now found all vertices of the achievable points --- but the algorithm does not know this yet.
In particular, there are two faces (to the left and to the right of $\vec{p}^{(3)}$) where $U_3 \setminus L_3$ is still nonempty.
By using weight vectors orthogonal to these faces, we can tighten the overapproximation. 
With these new weight vectors, we find policies whose induced points coincide with previously found points.
This also proves that the faces of $L_3$ are Pareto-optimal, i.e., $U_k = L_k$ and there are no further achievable points, as shown in the figure below.
```{code-cell} python
:tags: [remove-input]
_ = explore_pareto(mdp, ["red", "blue"], [(1.0, 0.0), (0.0, 1.0), (0.5, 0.9), (0.1, 0.35), (0.4, 0.55)], figsize=(2, 2), legend="outside") 
```
````

````{prf:remark} On computing weighted reachability
The weighted reachability query can be expressed as a total (undiscounted) reward query on the unfolding.
Specifically, you get reward $w_i$ for every transition between an $S_{-i}$ and $S_{+i}$, using the [earlier definitions](#def:splusandsminus). 
While the total reward in general is infinite, it is not if the reward collectible in every MEC is zero.
As the reward, by construction, is only awarded upon the first entry of some target set, we can only finitely often pick up reward.
Therefore, the total reward is finite and computable via all standard approaches, see @DBLP:phd/dnb/Quatmann23[Section 4.1.3].

This also connects weighted reachability to goal memory: since the reward query is solved on the unfolded MDP, the optimal policy is memoryless on the unfolded states $(s, b)$ — which is exactly a goal-memory policy on the original MDP, conditioning on the bit vector $b$ of visited targets.
````

#### Variant 3: Convex Hull Value Iteration
```{attention}
Currently skipped.
```

## Tractable hyperproperties
```{attention}
Work in progress
```