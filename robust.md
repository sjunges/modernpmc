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
Let $\Interval$ denote the set of intervals with rational lower and upper bounds.
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

We can define the reachability probability in an interval MDP:
```

```

A key result for interval MDPs is the following:
```{prf:theorem}

```
