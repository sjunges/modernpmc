# Robust MDPs

## What are robust MDPs?
Robust MDPs, generally, are often used to refer to MDPs with an unknown transition function.
The literature largely focuses on _s-a-rectangular_ MDPs.
These allow a slightly more convenient notation.
```{prf:definition} S-A rectangular Robust MDP
A (s-a-rectangular) robust MDP is a tuple $$ \langle S, A, \delta \rangle $$ where $S$ are states $A$ are actions as before,
and $$\delta\colon S \times A \nrightarrow 2^\Distr{S}$$, i.e., where the transition relation maps to a set of distributions over successors.
```
This definition of a robust MDP implies _s-a-rectangularity_:
The sets for each state-action pair are defined independently.
For robust MDPs beyond s-a-rectangularity, we refer to parametric MDPs.

Different robust MDPs make different assumptions about the shape and the way the uncertainty is resolved.
- the shapes of $\delta(s,a)$, i.e., as balls, simplices, etc.
- whether the uncertainty gets resolved statically, before execution, or dynamically, during execution. 

Interval MDPs are prominent robust MDPs.
Let $\Intervals$ denote the set of intervals with rational lower and upper bounds.
```{prf:definition} Interval MDP
An interval MDP is a tuple $$ \langle S, A, \delta \rangle $$ where $S$ are states $A$ are actions as before,
and $$\delta\colon S \times A \nrightarrow S \rightarrow \Intervals$$, i.e., where every transition is labelled with an interval.
```
```{prf:remark} 
Alternative definitions provide $\delta$ as a tuple of functions, one for the lower bound and one for the upper bound.
Both definitions are equivalent.
```
Interval MDPs can be interpreted with static and dynamic uncertainty.
Consider a path through the system where a state $s$ is visited twice and the action $a$ is picked every time.
Are the successor states governed by the same distribution in the set $\delta(s,a)$ or may the successor state be drawn from two different distributions?
The former is called static uncertainty, the second is called dynamic uncertainty.
For computing optimal reachability probabilities, both types of uncertainty coincide, but this is not true for many other properties.

When using the static interpretation, it is natural to think about the set of MDPs induced by an interval MDP.
```{prf:definition} Induced MDP and generated set
Given an iMDP $\imdp = \langle S, A, \delta \rangle$, 
an MDP $\mdp = \langle S, A, \delta' \rangle$ is induced by $\imdp$, if for all $s \in S, a \in A$
1. $\delta'(s,a)$ defined if $\delta(s,a)$ defined, and
2. for all $s' \in S$, $\delta'(s,a)(s') \in \delta(s,a)(s')$.
We denote the set of induced MDPs as 
$$ \generatorint{\imdp} = \{ \mdp \mid \mdp induced by \imdp \} $$
and also call this the (statically) generated set. 
```

Using the static interpretation, we can also define the reachability probability in an interval MDP:
```

```

A key result for interval MDPs is the following:
```{prf:theorem}
For any interval MDP with set of policies $\Policies$, it holds that:
$$ \max_{\pi \in \Policies} \min_{\mdp \in \generatorint{\imdp}} \pr^\pi_\mdp(\lozenge) = \min_{\mdp \in \generatorint{\imdp}} \max_{\pi \in \Policies}  \pr^\pi_\mdp(\lozenge)$$ 
```

## Robust value iteration

Robust value iteration for interval MDPs generalises the standard Bellman operator: 
at each state-action pair, rather than using a fixed transition distribution, nature picks the distribution within the interval that is most adversarial.
The resulting per-action expected value is computed by a _greedy corner-point_ algorithm that shifts remaining probability mass towards the worst-case successor.

````{prf:example}
Consider an interval MDP with four states: $s_0$ (initial), $s_1$, $\mathit{target}$, and $\mathit{sink}$.

From $s_0$ the agent chooses between two actions:
- **cautious**: reaches $\mathit{target}$ with probability in $[0.5, 0.7]$ and $\mathit{sink}$ with $[0.3, 0.5]$.
- **bold**: first reaches intermediate state $s_1$ with probability in $[0.7, 0.9]$ and $\mathit{sink}$ with $[0.1, 0.3]$.

From $s_1$ there is one action reaching $\mathit{target}$ with $[0.6, 0.9]$ and $\mathit{sink}$ with $[0.1, 0.4]$.

```{code-cell} python
:tags: [remove-input, remove-output]
import stormvogel.model as sv_model
import stormvogel.bird as bird
from stormvogel.model import Interval, ModelType
import stormvogel.teaching.bellman as bellman

(S0, S1, TARGET, SINK) = range(4)

def _available_actions(s):
    return ["cautious", "bold"] if s == S0 else ["go"]

def _delta(s, act):
    match s:
        case 0:
            if act == "cautious":
                return [(Interval(0.5, 0.7), TARGET), (Interval(0.3, 0.5), SINK)]
            else:
                return [(Interval(0.7, 0.9), S1), (Interval(0.1, 0.3), SINK)]
        case 1:
            return [(Interval(0.6, 0.9), TARGET), (Interval(0.1, 0.4), SINK)]
        case 2:
            return [(1.0, TARGET)]
        case 3:
            return [(1.0, SINK)]

imdp = bird.build_bird(
    _delta,
    available_actions=_available_actions,
    init=S0,
    labels=lambda s: ["target"] if s == TARGET else [],
    modeltype=ModelType.MDP,
    friendly_names=lambda s: ["s0", "s1", "target", "sink"][s],
)
```
```{code-cell} python
:tags: [remove-input]
sv.to_dot.plot_model_pydot(imdp)
```

**Robust value iteration** (nature is adversarial — it minimises the agent's expected value):

```{code-cell} python
:tags: [remove-input]
op_robust = bellman.make_operator_robust_maxreachprob(imdp, "target")
vi_robust = bellman.VI(op_robust, bellman.zero_value(imdp))
bellman.visualise_iterations([vi_robust.step() for _ in range(4)])
```

Under adversarial nature, **cautious** is optimal at $s_0$ with value $0.5$: nature would push the bold action towards $\mathit{sink}$, yielding only $0.7 \times 0.6 = 0.42$.

**Cooperative value iteration** (nature is optimistic — it maximises the agent's expected value):

```{code-cell} python
:tags: [remove-input]
op_coop = bellman.make_operator_coop_maxreachprob(imdp, "target")
vi_coop = bellman.VI(op_coop, bellman.zero_value(imdp))
bellman.visualise_iterations([vi_coop.step() for _ in range(4)])
```

Under optimistic nature, **bold** is optimal with value $0.81$: nature pushes the bold action towards $s_1$, and then towards $\mathit{target}$ from $s_1$, yielding $0.9 \times 0.9 = 0.81 > 0.7$.

The two results bound the true achievable value for any fixed MDP in the generated set.
````

