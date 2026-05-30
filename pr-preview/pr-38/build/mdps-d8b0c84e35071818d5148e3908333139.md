---
numbering:
  heading_1: true
  heading_2: true
  heading_3: true
  equations: false
  figures: true

kernelspec:
  name: python3
  display_name: Python 3
---

# Markov Decision Processes

```{code-cell} python
:tags:[remove-input]
import stormpy
import sympy
import stormvogel as sv
import stormvogel.teaching as teach
import stormvogel.bird as bird
import stormvogel.to_dot
sympy.init_printing()
from IPython.display import Math
import stormvogel.teaching.bellman as bellman

def create_mdp_one():
    def _available_actions(s):
        if s == 0:
            return ["a", "b"]
        else:
            return ["a"]

    def _delta(s,act):
        if s == 0 and act == "b":
            return [(0.5, 1), (0.5, 2)]
        else:
            return [(1,s)]

    def _labels(s):
        if s == 1:
            return ["target"]
        return []
        
    def _friendly_name(s):
        return "s"+str(s)

    return bird.build_bird(
        _delta, available_actions=_available_actions, init=0, labels=_labels, modeltype=sv.ModelType.MDP, friendly_names=_friendly_name
    )
mdp_one = create_mdp_one()


def create_mdp_parker():
    def _available_actions(s):
        if s in [0,3]:
            return ["a", "b"]
        return ["a"]

    def _delta(s, act):
        match s:
            case 0:
                match act:
                    case "a":
                        return [(0.25, 0), (0.5, 2), (0.25, 3)]
                    case "b":
                        return [(1,1)]
            case 1:
                return [(0.1, 0), (0.5, 1), (0.4, 2)]
            case 2: 
                return s
            case 3:
                match act:
                    case "a":
                        return 2
                    case "b":
                        return 3

    def _labels(s):
        if s == 2:
            return ["target"]
        return []
        
    def _friendly_name(s):
        return "s"+str(s)
    
    return bird.build_bird(
        _delta, available_actions=_available_actions, init=0, labels=_labels, modeltype=sv.ModelType.MDP, friendly_names=_friendly_name
    )
        
mdp_parker = create_mdp_parker()


def create_mdp_pi_max():
    # States: 0=s0 (initial), 1=s1, 2=s2, 3=s3, 4=t (target), 5=sink
    def _available_actions(s):
        return ["a", "b"] if s in [0, 1] else ["a"]

    def _delta(s, act):
        match s:
            case 0:
                match act:
                    case "a": return 1
                    case "b": return 2
            case 1:
                match act:
                    case "a": return [(1/2, 4), (1/2, 5)]
                    case "b": return 3
            case 2: return [(2/3, 4), (1/3, 5)]
            case 3: return [(3/4, 4), (1/4, 5)]
            case 4: return 4
            case 5: return 5

    def _labels(s):
        if s == 4:
            return ["target"]
        return []

    def _friendly_name(s):
        return ["s0", "s1", "s2", "s3", "t", "sink"][s]

    return bird.build_bird(
        _delta, available_actions=_available_actions, init=0,
        labels=_labels, modeltype=sv.ModelType.MDP, friendly_names=_friendly_name
    )

mdp_pi_max = create_mdp_pi_max()
```

# What are Markov decision processes?

Markov decision processes are a state-based model that combine nondeterministic action choices with probabilistic action outcomes.

## Formal definitions
````{prf:definition} Markov Decision Process
An _MDP_ $\mdp$ is a tuple

```{math}
:enumerated: false
\mdptuple
```

where

- $S$ is a nonempty set of _states_,
- $A$ is a nonempty set of _actions_,
- $\delta \colon S \times A \nrightarrow \Distr{S}$ is a _partial transition function_.
````

Additionally, we often assume the existence of a unique _initial state_ $\sinit$.
Later, we extend MDPs with _atomic propositions_, _target states_, and _reward functions_.

````{prf:definition} Enabled actions
For any state $s$, the set of _enabled actions_ is

```{math}
\EnAct{s} = \{ a \mid \delta(s,a) \neq \bot \}.
```
````

We use the notion of a _choice_ to denote a state–action pair
$\langle s,a \rangle$ where $a \in \EnAct{s}$.

Markov chains are MDPs with exactly one choice per state. We can then omit actions and write 
a Markov chain as a tuple $\langle S, \delta \rangle$, potentially extended by initial states.

For any choice $\langle s,a\rangle$, we may write $\delta(s,a,s')$ to denote $\delta(s,a)(s')$.

```{warning} Assumptions
Unless stated otherwise, we assume

- $S$ and $A$ are finite,
- there are no _deadlocks_: $\EnAct{s} \neq \emptyset$, and
- all probabilities $\delta(s,a)(s')$ are rational.
```

```{prf:definition} Path
A _path_ is a sequence
$
s_0 a_0 s_1 \dots \in (S \times A)^* \times S
$
such that for all $i$:
- $\langle s_i,a_i\rangle$ is a choice, and
- $\delta(s_i,a_i)(s_{i+1}) > 0$.

The set of all paths is denoted $\Paths^\mdp$. We drop $\mdp$ whenever it is clear from the context.
We write $\Paths^\mdp(s)$ for the set of paths with $s_0 = s$, and $\Paths(s,T)$ for paths that start in $s$ and end in $T$.
```
For a finite path
$
\xi = s_0 a_0 s_1 \dots s_n
$ 
we write $\last{\xi}$ for the final state $s_n$.

For an infinite path
$
\xi = s_0 a_0 s_1 a_1 s_2 \dots,
$
we define $$ \infinite{\xi} = \{ \langle s, a \rangle  \mid |\{ i \mid \langle s_i, a_i \rangle = \langle s, a \rangle \}| = \infty \} $$
as the choices that are made infinitely often along a path.

### Graph structure
(def:mdp:sinkstate)=
A _sink state_ is a state $s$ with $\delta(s,a)(s) = 1$ for all $a \in \EnAct{s}$.

(def:mdp:acyclic)= 
An MDP is _acyclic_ if along every infinite path, the only states visited infinitely often are sink states.

### Policies
Policies resolve nondeterminism.
```{note}
In the literature, different words are used. In the classical model checking literature, the word _scheduler_ is preferred, 
in game theory, the word _strategy_ is preferred, and in control theory, policies are often called _controllers_.
```

````{prf:definition} Policy
For an MDP $\mdp$, a _policy_ is a function

```{math}
:enumerated: false
\pi \colon \Paths^\mdp \rightarrow \Distr{A}.
```

- A policy $\pi$ is _memoryless_ if
$
\pi(\xi) = \pi(\xi')
$
for all paths with the same last state.
- A policy is _deterministic_ if every distribution $\pi(\xi)$ is Dirac.
````

We denote the set of all policies by $\Policies$. 
We use $\MdPolicies$ for the memoryless deterministic policies and $\MrPolicies$ for the memoryless (randomising) policies.
Memoryless deterministic policies can be written as
$
\pi \colon S \rightarrow A,
$
memoryless randomising as
$
\pi \colon S \rightarrow \Distr{A}.
$

Between the two extremes — general history-dependent policies and memoryless policies — lies the class of _finite-memory policies_, representable by a _finite-state controller_ (FSC).

```{prf:definition} Finite-state controller
:label: def:fsc
A _finite-state controller_ (FSC) is a tuple $\langle Q, q_0, \delta_Q, \sigma \rangle$ with
- a finite set of memory states $Q$ and initial memory state $q_0 \in Q$,
- a memory-update function $\delta_Q \colon Q \times S \times A \to \Distr{Q}$, and
- an action-selection function $\sigma \colon Q \times S \to \Distr{A}$.

Starting in memory state $q_0$, at each step the FSC selects action $a \sim \sigma(q, s)$ in MDP state $s$ with current memory $q$, then samples the next memory state $q' \sim \delta_Q(q, s, a)$.
Memoryless policies are the special case $|Q| = 1$.
An FSC is _deterministic_ if both $\sigma$ and $\delta_Q$ are Dirac; it is _action-independent_ if $\delta_Q(q, s, a) = \delta_Q(q, s, a')$ for all $a, a'$.
```

An FSC $\fsc$ induces a history-dependent policy $\pi_\fsc \colon \Paths^\mdp \to \Distr{A}$ as follows.
The _memory trace_ starts at $q(\xi_0) = q_0$, where $\xi_0 = s_0$ is the length-zero path prefix.
After executing action $a$ in MDP state $\last(\xi)$ with current memory $q(\xi)$, the next memory state is drawn from $\delta_Q(q(\xi), \last(\xi), a)$.
Then $\pi_\fsc(\xi) = \sigma(q(\xi), \last(\xi))$: the FSC selects actions based on the current memory and MDP state.

````{prf:example}
Consider the FSC with $Q = \{q_0, q_1\}$, actions $\{\alpha, \beta\}$, action selection $\sigma(q_0, s) = \alpha$ and $\sigma(q_1, s) = \beta$ for all $s$, and memory update $\delta_Q(q_i, s, a) = q_{1-i}$ (toggle on every step, regardless of state or action).
The memory trace and resulting policy choices for a path $s_0 \alpha s_1 \beta s_0 \alpha s_2$ are:

| Path prefix $\xi$ | $q(\xi)$ | $\pi_\fsc(\xi) = \sigma(q(\xi), \last(\xi))$ |
|---|---|---|
| $s_0$ | $q_0$ | $\alpha$ |
| $s_0 \alpha s_1$ | $\delta_Q(q_0, s_0, \alpha) = q_1$ | $\beta$ |
| $s_0 \alpha s_1 \beta s_0$ | $\delta_Q(q_1, s_1, \beta) = q_0$ | $\alpha$ |
| $s_0 \alpha s_1 \beta s_0 \alpha s_2$ | $\delta_Q(q_0, s_0, \alpha) = q_1$ | $\beta$ |

The action alternates on every step regardless of which MDP states are visited.
No memoryless policy can reproduce this behaviour, since a memoryless policy must choose the same action distribution at $s$ on every visit.
````

Finite-memory policies become essential when the optimal action depends on history — as is the case for @sec:dfa and for @chap:multiobjective.

We furthermore define the set of paths under a policy $\Paths^\pi$ as the set of paths that are consistent with a policy $\pi$. 


### Induced Markov Chains

Once a policy resolves nondeterminism in an MDP, the behaviour becomes purely probabilistic and can thus be captured by a Markov chain.

````{prf:definition} Induced Markov chain (general policies)
Given MDP $\mdp$ and policy $\pi$, the _induced Markov chain_
$\mdp[\pi]$
is
$
\langle \Paths^\mdp , \delta^\pi \rangle
$
with

```{math}
:enumerated: false
\delta^\pi(\xi)(\xi \cdot a s)
=
\pi(\xi)(a)\,\delta(\last{\xi},a)(s).
```
````
```{prf:remark} Infinite Markov chain
The above Markov chain is infinite, but reachability probabilities remain well defined.
```
It is useful to consider a special case for memoryless (possibly randomised) policies.
`````{prf:definition} Induced Markov chain (memoryless policies)
For a memoryless policy $\pi$, the induced Markov chain is $\mdp[\pi] =
\langle S , \delta^\pi \rangle$
where
```{math}
:enumerated: false
\delta^\pi(s)(s') = \sum_a \pi(s)(a)\delta(s,a)(s'). 
```
`````

````{prf:definition} Induced Markov chain (FSC)
:label: def:mdp:induced-fsc
For an FSC $\fsc = \langle Q, q_0, \delta_Q, \sigma \rangle$, the _induced Markov chain_ is
$\mdp[\fsc] = \langle S \times Q,\, \delta^{\fsc} \rangle$
with initial state $(\sinit, q_0)$ and transitions
$$
\delta^{\fsc}((s,q))((s',q')) = \sum_a \sigma(q,s)(a)\,\delta(s,a)(s') \cdot \delta_Q(q,s,a)(q').
$$
````

```{prf:remark} Three definitions, one concept
All three induced Markov chains are consistent (up-to some technicality): the general construction (over paths) is bisimilar to the FSC construction (over $S \times Q$), which in turn reduces to the memoryless construction (over $S$) when $|Q| = 1$ — since the single memory state $q_0$ carries no information and $(s, q_0)$ collapses to $s$.
```

````{prf:definition}
:label:def:mdps:generatorinduced
Given an MDP $\mdp$ and a set of policies $\Pi$, we define $$ \generator{\mdp}{\Pi} = \{ \mdp[\pi] \mid \pi \in \Pi \}. $$
We define $$ \generatormd{\mdp} = \generator{\mdp}{\MdPolicies} \text{ and } \generatormr{\mdp} = \generator{\mdp}{\MrPolicies}$$
````

# Reachability probabilities

For a policy $\pi$, the _reachability probability_
$
\pr^\pi_\mdp(s \models \lozenge T)
$
denotes the probability of eventually reaching a target state $T$ from a state $s$. 
The probability $\pr^\pi_\mdp(s \models \lozenge T)$ is defined via the induced MC $\mdp[\pi]$.
We are mostly interested in this probability from the initial state $\sinit$, in which case we simplify notation to
$\pr^\pi_\mdp(\lozenge T)$.

```{admonition} Problem: Standard verification problem for reachability probabilities
Given MDP $\mdp$, a _threshold_ $\lambda \in \mathbb{Q}$ and $\bowtie \in \{\leq,\geq\}$, decide whether

$$
\pr^\pi_\mdp(\lozenge T) \bowtie \lambda
$$

holds for **all** policies.
```

```{admonition} Problem: Standard planning problem for reachability probabilities
Given MDP $\mdp$, $\lambda \in \mathbb{Q}$ and $\bowtie$, compute **some** policy $\pi$ such that

$$
\pr^\pi_\mdp(\lozenge T) \bowtie \lambda .
$$
```

(sec:qualitative)=
## Qualitative verification

Qualitative verification refers to the setting where the threshold $\lambda$ is either zero or one.
As we will see, qualitative verification can be solved without reference to the precise probabilities in the MDP.
Assume fixed $\mdp$ and target set $T$.

````{prf:example} Qualitative verification - running example 
:label:ex:mdp:qualitative
Throughout this section, we use the following MDP as a running example.
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.qualitative_mdp import spos, sposmin, smaxas, run_and_collect, visualise_iterations

mdp_qual = sv.model.new_mdp()
s0_q  = mdp_qual.initial_state;  s0_q.set_friendly_name("s0")
s1_q  = mdp_qual.new_state(friendly_name="s1")
s2_q  = mdp_qual.new_state(friendly_name="s2")
s3_q  = mdp_qual.new_state(friendly_name="s3")
s4_q  = mdp_qual.new_state(friendly_name="s4")
st_q  = mdp_qual.new_state("target", friendly_name="t")
sk_q  = mdp_qual.new_state("sink",   friendly_name="sink")

act_a_q = mdp_qual.new_action("a")
act_b_q = mdp_qual.new_action("b")

s0_q.set_choices({act_a_q: [(1.0, s1_q)],                act_b_q: [(1.0, sk_q)]})
s1_q.set_choices({act_a_q: [(0.5, st_q), (0.5, s2_q)],   act_b_q: [(1.0, st_q)]})
s2_q.set_choices({act_a_q: [(1.0, s3_q)],                act_b_q: [(1.0, sk_q)]})
s3_q.set_choices({act_a_q: [(1.0, st_q)]})
s4_q.set_choices({act_a_q: [(0.3, st_q), (0.7, sk_q)]})
mdp_qual.add_self_loops()
target_states_q = list(mdp_qual.state_labels["target"])
sv.to_dot.plot_model_pydot(mdp_qual)
```
````

### Possible reachability (max)
One of the simplest questions we can ask is to find the set of states that can reach the targets with positive probability.

More formally, we denote this set of states  $$\Spos = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) > 0 \}$$, i.e., the set of states where the _maximum_ probability over all policies is positive. 
It holds that 

$\Spos = S \setminus \Smaxzero$
using
```{math}
\Smaxzero = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 0 \}.
```
The notation contrasts with $\Szero = \{ \forall \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 0 \}$ where the _minimum_ probability over all policies is zero and which we discuss below.

```{prf:theorem}
For every state $s$,

$$
s \in \Spos
\quad\text{iff}\quad
\exists \xi \in \Paths(s,T).
$$
```

The main consequence is that computing $\Spos$ reduces to graph reachability.
We omit a technical proof of the theorem and focus on the intuition:
- If there is a path in the MDP, then there is a policy that takes exactly the choices in that path. Pick such a policy. In the induced Markov chain, the same path exists, and thus, the probability is positive. 
- If there is no such path in the MDP, then a policy cannot make choices that lead to a path in the induced Markov chain.


This can be computed as a least fixpoint. Specifically, we define

$$
\Psi_{{\max}>0}\colon 2^S \rightarrow 2^S
$$
such that
$$
\Psi_{{\max}>0}(X)=
\{ s \in S \mid \exists a \in \EnAct{s}.\, \exists s' \in X.\, \delta(s,a)(s') > 0 \} \cup T
$$
The operator is monotonic and the lattice is finite.

```{prf:theorem}
$\lfp{\Psi_{{\max}>0}} = \Spos$.
```

````{prf:example}
Consider @ex:mdp:qualitative.
Starting from $X_0 = T$, the operator $\Psi_{\max>0}$ adds all states with any action reaching the current set in one step.
```{code-cell} python
:tags: [remove-input]
pass
visualise_iterations(run_and_collect(spos(mdp_qual, target_states_q)), mdp_qual)
```
````

### Possible reachability (min)
We now study computing
$$ \Sposmin = \{ \forall \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) > 0 \}, $$
that is, the complement of $\Szero$. Concretely, we will use that each policy reaches the target with positive probability iff it has a path to the target of length $n$.
This can be computed as a least fixpoint. Specifically, we define 

$$
\Psi_{{\min}>0}\colon 2^S \rightarrow 2^S
$$
such that
$$
\Psi_{{\min}>0}(X)=
\{ s \in S \mid \text{for all } a \in \EnAct{s} \exists s' \in X. \delta(s,a)(s') > 0  \} \cup T
$$
The operator is monotonic and the lattice is finite. 

```{prf:theorem}
$\lfp{\Psi_{{\min}>0}} = \Sposmin$.
```

````{prf:example}
Consider @ex:mdp:qualitative.
Starting from $X_0 = T$, the operator $\Psi_{\min>0}$ adds states where every action has at least one successor already in the current set.
```{code-cell} python
:tags: [remove-input]
visualise_iterations(run_and_collect(sposmin(mdp_qual, target_states_q)), mdp_qual)
```
````

### Almost-sure reachability (max)
We are also interested in computing the set of states from which it is possible to ensure that we almost-surely reach the target states.

More formally, we denote this set of states $$\Smaxas = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 1 \}.$$

We use a recursive equation to characterise this set. 
If $s \in T$, then clearly $s \in \Smaxas$. 
Otherwise, for $s \not \in T$:  $$s \in \Smaxas \text{\quad iff \quad} \exists a \in \EnAct{s}. \forall s' \in \supp{\delta(s,a)}. s' \in \Smaxas. $$

To compute the set $\Smaxas$, we want to provide an iterative procedure, which operates on sets of states.
Specifically, we define the operator

$$
\Psi_{{\max}=1}\colon 2^S \rightarrow 2^S
$$
such that
$$
\Psi_{{\max}=1}(X)=
\{ s \in \Spos \mid
\exists a\in\EnAct{s}.
\forall s'\in\supp{\delta(s,a)}.
s'\in X \}\cup T
$$
The operator is monotonic and the lattice is finite. 

```{prf:theorem}
$\gfp{\Psi_{{\max}=1}} = \Smaxas$.
```

Restricting the operator's co-domain to $\Spos$ is essential.
Without it, $S$ itself is always a fixed point: any non-target state with a self-loop has an action whose only successor is itself, so it is never removed.
By restricting to $\Spos$, states from which $T$ is unreachable are excluded before the GFP begins.

An equivalent formulation (see @BK08, Algorithm 45) operates at _action_ granularity: actions whose support intersects $S \setminus X$ are removed from the MDP, and a state is only removed once all its actions are gone.
The two formulations agree because the GFP sequence is decreasing — once an action's successor leaves the current set, the successor never returns — so the set of invalidated actions grows monotonically, exactly as in the action-removal algorithm.
Our set-based formulation is more concise; the action-removal view is closer to an efficient implementation.

````{prf:example}
Consider @ex:mdp:qualitative. The GFP starts from $\Spos$ (states that can possibly reach $T$) and removes states that lack any action with all successors inside the current set.
```{code-cell} python
:tags: [remove-input]
visualise_iterations(run_and_collect(smaxas(mdp_qual, target_states_q)), mdp_qual)
```
````

It is important to note that extracting a policy from the set of states where a witness policy for almost-surely reaching $T$ exists does not make it trivial to extract this policy.
The policy surely has to select actions in every state such that we stay within the set from which a witnessing policy exists.
Additionally, to ensure that the target is indeed reached, we must avoid introducing SCCs in the induced MC.
A simple method is to compute a **randomised** policy: we assign positive probability to every action $a$ at $s \in \Smaxas$ whose support lies within $\Smaxas$.
Since $s \in \Smaxas$ guarantees at least one such action exists (by definition of the GFP operator $\Psi_{\max=1}$), this policy is well-defined.
Under it, no non-target strongly connected component within $\Smaxas$ is a bottom SCC, i.e., each SCC is left until we eventually reach the target. 
The right notion to reason about the extraction of deterministic policies is that of *maximal end-components* (see @sec:mecs).


### Almost-sure reachability (min)
Finally, we can also define the set of states $$\Sminas = \{ s \mid \forall \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 1 \}.$$


```{attention}
Not discussed in the lecture. 
```

### Relationships between the qualitative sets

The four sets form two inclusion chains with $\Sminas$ as the smallest and $\Spos$ as the largest:

$$\Sminas \subseteq \Sposmin \subseteq \Spos \qquad \text{and} \qquad \Sminas \subseteq \Smaxas \subseteq \Spos.$$

The middle sets $\Sposmin$ and $\Smaxas$ are **incomparable** in general: a state may lie in $\Sposmin \setminus \Smaxas$ (every policy reaches $T$ with positive probability, yet no policy guarantees probability~$1$) or in $\Smaxas \setminus \Sposmin$ (some policy reaches $T$ almost surely, yet another policy avoids $T$ entirely).

## Quantitative reachability

We are interested in computing the

* maximal reachability probability

$$
\sup_{\pi \in \Policies} \pr^\pi(s \models \lozenge T), 
$$

and the

* minimal reachability probability

$$
\inf_{\pi \in \Policies} \pr^\pi(s \models \lozenge T).
$$

```{note}
The use of minimal and maximal rather than infimum and supremum in the naming is adequate due to @thm:mdsuffices:minreachprob and @thm:mdsuffices:maxreachprob respectively.
```

While both cases are mostly symmetrical, the minimal reachability probability is a bit easier to compute.   

### Minimal reachability probability

```{prf:theorem} Bellman equations (MinReachProb) @BK08 [Thm 10.109]
:label: thm:bellmaneq:minreachprob
Given an MDP with states $S$. Consider variables $x_s$ for each $s \in S$.
The minimal reachability probability is the _unique_ solution to the following equation system
$$
x_s=
\begin{cases}
1 & s\in T \\
0 & s\in \Szero \\
\min_{a\in \EnAct{s}}
\sum_{s'}\delta(s,a,s')x_{s'}
& \text{otherwise}
\end{cases}
$$
These equations are called the Bellman equations _for minimal reachability probabilities_.
```

````{figure}
:label: fig:mdpparkervis
```{code-cell} python
:tags: [remove-input]
sv.to_dot.plot_model_pydot(mdp_parker)
```
Visualisation of an MDP to illustrate the [Bellman equations for MinReachProb](#thm:bellmaneq:minreachprob).
````

````{prf:example}
:label: ex:bellman:minreachparker
Consider the MDP from @fig:mdpparkervis. The Bellman equations for minimal reachability probability are:
```{code-cell} python
:tags: [remove-input]
equations = bellman.minreachprob(mdp_parker, "target")
Math(r"\\".join([sympy.latex(eq) for eq in equations]))
```
The equations have a unique solution.
````


```{hint} Bellman equations vs Bellman operators
We will later also talk about Bellman operators. 
At that point, we will clarify the difference.
```
```{hint} Different Bellman equations 
We will see various different Bellman equations. It is important to clarify which Bellman equations one is talking about.
```

We observe the following:
- If there is only one choice per state, the equation system trivially reduces to the linear equation system for Markov chains.
- Using the preprocessing with $\Sminas$, we could extend the set of states for which we can ensure $x_s = 1$.
- We must ensure that states that cannot reach the target have a zero probability.

```{prf:theorem} Memoryless policies suffice (MinReachProb)
:label: thm:mdsuffices:minreachprob
For any MDP and target set $T$:
1. For any state $s$ it holds that
$$ \inf_{\pi \in \Policies} \pr^\pi(s \models \lozenge T) = \min_{\pi \in \MdPolicies} \pr^\pi(s \models \lozenge T). $$
2. There exists a policy $\pi^{*}$ such that for all $s \in S$:
$$ \pr^{\pi^{*}}(s \models \lozenge T) = \min_{\pi \in \MdPolicies} \pr^\pi(s \models \lozenge T).$$
```
For any solution to the Bellman equations, it is simple to extract **a** witnessing memoryless deterministic policy $\pi^{*}$: One simply takes $$\pi^{*}(s) = \argmin_{a} \sum_{s'} \delta(s,a)(s') x_{s'}.$$

We highlight that there is a unique solution to the Bellman equations, but not a unique minimising policy. 
Furthermore, the theorem justifies talking about minimising policies for a target set, independently of a specific starting state.



### Maximal reachability probability
While most aspects of computing minimal and maximal reachability probabilities are analogous, the theorems about the Bellman equations differ for both cases.
```{prf:theorem} Bellman equations (MaxReachProb)
:label: thm:bellmaneq:maxreachprob
Given an MDP with states $S$. Consider variables $x_s$ for each $s \in S$.
The maximal reachability probability equals the _minimal_ solution

$$
x_s=
\begin{cases}
1 & s\in T \\
0 & s\in \Smaxzero \\
\max_{a\in \EnAct{s}}
\sum_{s'}\delta(s,a,s')x_{s'}
& \text{otherwise}
\end{cases}
$$
```

````{figure}
:label: fig:mdponevis
```{code-cell} python
:tags: [remove-input]
sv.to_dot.plot_model_pydot(mdp_one)
```
Visualisation of an MDP with no unique solution for [Bellman equations for MaxReachProb](#thm:bellmaneq:maxreachprob).
````

````{prf:example} 
:label: ex:maxreachprobnotunique
Consider the MDP in @fig:mdponevis. The [Bellman equations for MaxReachProb](#thm:bellmaneq:maxreachprob) are:

```{code-cell} python
:label:eq1
:tags: [remove-input]
equations = bellman.maxreachprob(mdp_one, "target")
Math(r"\\".join([sympy.latex(eq) for eq in equations]))
```
In particular, any assignment to $x_0 \geq 0.5$ is part of a valid solution. However, @thm:bellmaneq:maxreachprob clarifies that only $x=0.5$ is a valid solution. 
````
In general, the problem we observe here is that there are sets of state-action pairs (choices) under which a policy can stay indefinitely. 
If the states of these choices do not include target states, then staying there forever means that those states have reachability probability $0$,
yet the Bellman equations do not enforce that these states must be assigned to zero. 
In @sec:mecs, these sets of state-action pairs are called end-components and we show that (roughly) in MDPs without such end-components, the [solutions to the Bellman equations are unique](#thm:mdp:mecfreemaxreach).


```{prf:theorem} Memoryless policies suffice (MaxReachProb)
:label: thm:mdsuffices:maxreachprob
For any MDP and target set $T$:
1. For any state $s$ it holds that
$$\sup_{\pi \in \Policies} \pr^\pi(\lozenge T) = \max_{\pi \in \MdPolicies} \pr^\pi(\lozenge T) $$
2. There exists a policy $\pi^{*}$ such that for all $s \in S$:
$$ \pr^{\pi^{*}}(s \models \lozenge T) = \max_{\pi \in \MdPolicies} \pr^\pi(s \models \lozenge T).$$
```
As there is no unique solution to the Bellman equations when maximising, extracting an optimal policy is harder. 
In particular, while any optimal policy $\pi$ must satisfy $$\pi(s) = \argmax_{a} \sum_{s'} \delta(s,a)(s') x_{s'},$$ this condition is not sufficient. 
```{prf:example}
Consider @ex:maxreachprobnotunique. Both actions in $s_0$ will yield value $0.5$.
```
Instead, we must find a policy that makes progress towards the targets. 
A formal construction is given in @BK08 [Lemma 10.102], the idea is as follows.
First, we preprocess the MDP: we only consider actions in $A^{\max}(s)$, the set of actions that are consistent with the maximum in the Bellman equations.
All other actions can be removed from the MDP without affecting the maximal reachability probability. 
On this MDP, we then define the distance to the target as the length of the shortest path to the target.
This path is bounded in every state that has a positive probability to reach the target. 
A memoryless deterministic policy $\pi$ is then defined inductively based on this lenght $n$:
for $s \in T$ (i.e., $n=0$) the policy choice does not matter; for $n > 0$, choose any action $a \in A^{\max}(s)$ with positive probability to reach some state $s'$ whose distance is smaller than that of $s$.
This intuitively ensures that with some probability, we make progress towards the target, and more formally ensures that we do not create strongly connected components in the induced DTMC.



### The notion of value
To simplify the exposition, it is customary to merge discussions about minimal and maximal reachability probabilities to a fixed set of target states $T$.
If we know from context that we want to compute the (minimal or maximal) reachability probability from state $s$ for a set of target states $T$, we can call
- the probability induced by a policy _the value of the policy_ (from state $s$),
- the minimal (resp. maximal) probability from a state $s$ _the value of $s$_,
- the value of the initial state is the _value of the MDP_.
- We use $V^\pi_\mdp(s)$ to define the value induced by a policy and $V^{*}_\mdp(s)$ to define the value of a state. We omit $\mdp$ whenever possible.
```{danger} Implicit notation
The use of values of MDPs etc is always contextual, and this is never clear from the notation. We sometimes write $V^{min}$ or $V^{max}$ instead of $V^{*}$ to be more explicit.
```
```{note} Value beyond reachability probabilities
Later, we will also use value for other properties, including (discounted or total) expected rewards, and the probabilities for temporal properties.
```

(sec:mecs)=
## Maximal end components
End-components are sub-MDPs induced by a set of choices (state-action pairs) where a policy can visit every state and stay forever.
```{prf:definition}
An end-component for an MDP $\mdp = \mdptuple$	is a set $X \subseteq S \times A$ such that:
- $X$  induces the  submdp  $\mdp[X]  = \langle S', A',  \delta' \rangle$ with 
    * $S' = \{ s \mid \langle s, a \rangle \in X \}$,
    * $A' = \{ a \mid \langle s, a \rangle \in X \}$, and
    * $\langle s, a \rangle \not\in X$ implies $\delta'(s, a) = \bot $.
- (The graph underlying) $\mdp[X]$ is a strongly connected component[^scc].	
```
[^scc]: In a strongly connected component, every state can reach every state

```{prf:theorem} @BK08 [Theorem 10.120]
:label: thm:finallyendcomponent
For any policy $\pi$ and any state $s$:
$$ \pr_s^\pi(\{ \xi \in \Paths^\pi(s) \mid \infinite{\xi} \text{ is an EC}   \}) = 1 $$
```
That is, the probability that the choices visited infinitely often from any state onwards form an EC is one. 
In particular, this means that there are still paths that do not eventually end up in an EC, as there are paths that take any loop infinitely often.

```{prf:remark} MECs and qualitative reachability
:label: rem:mec-qualitative
The theorem connects end-components to the qualitative sets from @sec:qualitative.
A non-target EC is any EC whose states are disjoint from $T$; this includes non-target sink states (trivial MECs) as a special case.
Under any policy, every run eventually gets trapped in some EC with probability~$1$.
If that EC is non-target, the run never reaches $T$.
Hence a state $s$ lies in $\Spos \setminus \Smaxas$ precisely when some policy reaches $T$ positively, yet every policy also assigns positive probability to being trapped in a non-target EC.
Conversely, $s \in \Smaxas$ means a policy exists whose runs are trapped only in target-containing ECs.
```

End-components can overlap and can contain other end-components. 
It is often helpful to consider __maximal end components__ (MECs): 
Maximal end components are end components not properly contained in any other end component.
MECs cannot overlap. 
MECs can be detected with an efficient graph algorithm @BK08 [Algorithm 47]. 

````{prf:example}
Consider the MDP with six states shown below.
```{code-cell} python
:tags: [remove-input, remove-output]
import stormvogel.examples as examples
from stormvogel.stormpy_utils.mec import detect_mecs, eliminate_mecs
mdp_ec = examples.create_mixed_mec_mdp()
```
```{code-cell} python
:tags: [remove-input]
sv.to_dot.plot_model_pydot(mdp_ec)
```
```{code-cell} python
:tags: [remove-input]
mecs = detect_mecs(mdp_ec)
[f"MEC {i}: {sorted(s.friendly_name for s in mec)}" for i, mec in enumerate(mecs)]
```
The output lists the _state projection_ of each MEC (the states whose choices belong to the MEC), not the full set of state-action pairs.

States $s_1$ and $s_2$ have a cycle, but every action has positive probability of escaping to $s_5$.
No policy can keep the process inside $\{s_1, s_2\}$ forever, so there is no end-component over these states.

State $s_5$ is absorbing (a self-loop); the single state-action pair $\{(s_5, \mathit{self})\}$ forms a _trivial_ MEC.

States $s_3$ and $s_4$ each have a "loop" action that transitions to the other state with probability 1.
The state-action pairs $\{(s_3, \mathit{loop}), (s_4, \mathit{loop})\}$ form a strongly connected sub-MDP that is closed under the chosen actions, making them a _non-trivial_ MEC.
The fact that $s_3$ also has an "escape" action to $s_1$, and $s_4$ a "self" action, does not matter: a policy _exists_ that stays inside forever, and that is sufficient.
The "self" action of $s_4$ alone yields another end-component: $\{(s_4, \mathit{self})\}$.
Since $\{(s_4, \mathit{self})\}$ is properly contained in the MEC $\{(s_3, \mathit{loop}), (s_4, \mathit{loop})\}$, it is an end-component that is not a maximal end-component.

````

(def:mdp:trivialmec)=
We note that sink states with their self-loops are always MECs: We call these _trivial_ MECs.

(def:mdp:mecfree)= 
Analogously to the notion of an [acyclic MDP](#def:mdp:acyclic),
we call an MDP _MEC-free_, if all MECs are [trivial](#def:mdp:trivialmec).

```{note}
Importantly, an MEC-free MDP has MECs, just like an acyclic MDP has cycles.
```

```{prf:theorem} 
:label: thm:mdp:mecfreemaxreach
If the MDP in @thm:bellmaneq:maxreachprob is MEC-free, then the equations have a unique solution.
```
The statement indeed follows directly from @thm:finallyendcomponent. 


% Add example of an MDP with cycles but without MECs.



(sec:eliminatemecs)=
### Maximal End-Component collapsing
MDPs can be transformed into a MEC-free MDP while preserving _either_ min or max reachability probabilities for a fixed target.
The transformation is called _MEC collapsing_. We only discuss the MEC collapsing for maximising. 

```{prf:definition} MEC collapsing
:label: def:mec:collapsing
Given an MDP $\mdp$ with MECs $E_1, \dots, E_k$ and associated state sets $S_1, \dots S_k$ the _collapsed MDP_ $\mdp'$ is obtained by:
- replacing each non-trivial MEC $E_i$ with a single representative state $\hat{E}_i$ carrying the union of labels of all states in $S_i$,
- redirecting every transition into any state in $S_i$ to $\hat{E}_i$, and
- For $\hat{E}_i$, add all choices that leave the MEC, i.e., for any $s \in S_i$ with $(s,a) \not\in E_i$, we add a new choice $(\hat{E}_i, a_s) = \delta(s,a)$.  
States outside any non-trivial MEC are unchanged.
```
The intuition behind the new actions $s_a$ leaving the representative of a MEC is 
that a policy can choose any (weighted combination) of the original state-action pairs to leave an MEC.
```{prf:theorem}
For any target set $T$ and any state $s$ not inside a non-trivial MEC, the maximal reachability probabilities $\pr^{\max}(s \models \lozenge T)$ are preserved by MEC collapsing.
```
````{prf:example}
We collapse the end components of the MDP from the previous example.
```{code-cell} python
:tags: [remove-input]
mdp_collapsed, state_map = eliminate_mecs(mdp_ec, remove_representative_selfloops=False)
sv.to_dot.plot_model_pydot(mdp_collapsed)
```
The non-trivial MEC $\{s_3, s_4\}$ is merged into one representative state.
The trivial MEC $\{s_5\}$ and the non-MEC states $s_0$, $s_1$, $s_2$ are preserved unchanged.
````


# Algorithms for reachability probabilities

## Policy enumeration
As memoryless deterministic policies suffice, we can compute minimal and maximal reachability probabilities by simply enumerating over all memoryless deterministic policies.
We note, however, that this is exponential in the number of states and therefore highly impractical.

## Policy iteration

Policy iteration (PI) avoids the exponential cost of policy enumeration.
Starting from an initial policy $\pi_0$, it alternates between two steps until $\pi_i = \pi_{i+1}$:

- **Policy evaluation**: compute $V^{\pi_i}$ by solving the linear system of equations induced by $\pi_i$.
- **Policy improvement**: set
  $$\pi_{i+1}(s) = \argmax_a \sum_{s'} \delta(s,a)(s') \cdot V^{\pi_i}(s'),$$
  updating an action only if it yields a strict improvement (replace $\argmax$ by $\argmin$ for minimisation).

For maximisation, any initial policy suffices.
For minimisation, care is needed: a state $s \in S \setminus \Sposmin$ has minimum reachability probability 0, but a naive initial policy may assign it a positive value and cause PI to diverge.
The fix is to start from a *proper* initial policy: for each $s \in S \setminus \Sposmin$, pick an action whose support stays within $S \setminus \Sposmin$.
Such an action always exists: $s \notin \Sposmin$ means exactly that some action has no successor in $\Sposmin$ (by the fixpoint characterisation of $\Psi_{\min>0}$).
Under a proper initial policy, $S \setminus \Sposmin$ is an absorbing set; policy evaluation naturally assigns those states value 0, and no improvement step ever moves out of that set.
```{prf:theorem}
Policy iteration terminates after finitely many steps and the resulting policy is optimal.
```
The termination argument has two parts.
First, the improvement step guarantees $V^{\pi_{i+1}}(s) \geq V^{\pi_i}(s)$ for every state $s$: each state either keeps its action (unchanged value) or switches to one with a strictly higher one-step Bellman value under $V^{\pi_i}$, and a standard inductive argument propagates these local improvements to the global values of the new policy.
Second, whenever $\pi_{i+1} \neq \pi_i$, the strict improvement at at least one state means $V^{\pi_{i+1}} \neq V^{\pi_i}$, so no value vector — and hence no policy — is ever revisited.
Since there are at most $\prod_{s \in S} |\EnAct{s}|$ memoryless deterministic policies, PI must terminate.
The termination condition $\pi_i = \pi_{i+1}$ then implies no action can be improved, which by the Bellman equations means the current policy is optimal.

````{prf:example}
:label: ex:pi:minreachparker
We run policy iteration for minimal reachability probability on @fig:mdpparkervis with $s_2$ as the target.
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.policy_iteration import PI, visualise_pi_iterations

pi = PI(mdp_parker, "target", minimize=True)
visualise_pi_iterations(pi)
```
PI converges in two steps.
The proper initial policy sends $s_3$ to action $b$ (self-loop within $S \setminus \Sposmin$), fixing its value to 0.
In step 0, $s_0$ takes action $b$ (pessimistic: goes to $s_1$), yielding value 1; improvement then switches $s_0$ to action $a$.
In step 1 the new policy is evaluated to give the exact minimal reachability probabilities, and no further improvement is possible.
````
````{prf:example}
:label: ex:pi:maxreachpimax
We run policy iteration for maximal reachability probability with $t$ as the target on the following MDP,
starting from the all-$a$ policy.
States $t$ (target) and $\mathit{sink}$ are absorbing.
```{code-cell} python
:tags: [remove-input]
sv.to_dot.plot_model_pydot(mdp_pi_max)
```
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.policy_iteration import PI, visualise_pi_iterations
import stormvogel.result as result

taken = {}
for s in mdp_pi_max.states:
    for action, branch in s.choices:
        if action.label == "a":
            taken[s] = action
            break
    else:
        taken[s] = list(s.choices)[0][0]
sched0 = result.Scheduler(mdp_pi_max, taken)

pi_max = PI(mdp_pi_max, "target", minimize=False, scheduler=sched0)
visualise_pi_iterations(pi_max)
```
PI converges in two steps, but a state switches action _twice_.

**Step 0.** 
Evaluation: $s_1$ reaches $t$ or $\mathit{sink}$ with equal probability, giving $V^{\pi_0}(s_1)=\tfrac{1}{2}$, while $s_2$ and $s_3$ reach $t$ with probability $\tfrac{2}{3}$ and $\tfrac{3}{4}$ respectively.
Improvement: $s_0$ switches to $b$ (going directly to $s_2$ with value $\tfrac{2}{3}$ beats going to $s_1$ with value $\tfrac{1}{2}$); $s_1$ switches to $b$ (going to $s_3$ with value $\tfrac{3}{4}$ beats the coin flip).

**Step 1.** We use $\pi_1 = \{s_0 \mapsto b,\, s_1 \mapsto b\}$
Evaluation : $s_1$ now evaluates to $\tfrac{3}{4}$ (via $s_3$), which exceeds $s_2$'s fixed value of $\tfrac{2}{3}$.
Improvement: $s_0$ switches _back_ to $a$ — routing through $s_1$ (and on to $s_3$) is now better than going directly to $s_2$.
$s_1$ keeps $b$ (no strict improvement available).

**Step 2.** No action at any state yields a strict improvement; PI terminates.

The example illustrates that individual action choices need not be monotone across PI steps: the value function improves monotonically, but which action is optimal can change as the values of other states are updated.
````
```{prf:remark}
Note that the discussion above assumes exact solving of the induced DTMC. Any non-exact solver breaks _all_ guarantees about the correctness of PI.
```


## Linear programming
The next algorithm simply reduces the problem of computing reachability probabilities to a linear programming problem. 
### Short recap
Linear programming refers to an optimisation problem (a mathematical program) of a particular form.
```{prf:definition} Linear programming
Given $n$ real-valued variables $x_1, \dots, x_n$  and constants $a_{ij}$, $b_i$, $c_j$, with $1 \leq i \leq m$ and $1 \leq j \leq n$, 
a linear program is of the form:
\begin{align*}  \text{Maximize}\quad & c_1 \cdot x_1 + \dots c_n \cdot x_n \\
 & \text{such that } \\
		& a_{11} \cdot x_1 + a_{12}  \cdot x_2 + \dots a_{1n}  \cdot x_n \leq b_1\\
		& \dots \\
		& a_{m1} \cdot x_1 + a_{m2}  \cdot x_2 + \dots a_{mn} \cdot x_n \leq b_m
\end{align*}
```
We can support minimisation and $\geq$ by adequately negating constants.
Linear programming is supported by a rich theory. Here, it is important to note that linear programming admits polynomial-time solutions and there is very mature (academic and commercial) tool support for solving linear programs.

### Reduction
We again consider first the minimal reachability probability and then the maximal reachability probability.
#### Minimal reachability probabilities
The essence of the [Bellman equations](#thm:bellmaneq:minreachprob) is that in every state, we can pick a minimising action.
We translate that and say that the value of a state must be smaller than the value induced by any action. It is thus necessarily at most the value of the minimal action.
We then maximise the value of every state to ensure that we meet the solution. 
```{prf:theorem} 
The minimal reachability probability is the unique solution to the following LP with variables $x_s$ for $s \in S$.
\begin{align*}
	\text{maximize}\quad & \sum_{s \in S} x_s & &   \\
	  & x_s = 1 & & \text{ for all $s \in T$} \\
	  & x_s = 0 & & \text{ for all $s \in \Szero$} \\
	  & x_s \geq \sum_{s'} \delta(s,a)(s')x_{s'} & & \text{for all other $s$ and all $a\in \EnAct{s}$}
\end{align*}
```
````{prf:example}
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.lp import lp_minreachprob
lp_minreachprob(mdp_parker, "target")
```
````

#### Maximal reachability probabilities
The adaptions are straightforward. What is notable is that the [Bellman equations](#thm:bellmaneq:maxreachprob) require that we find the smallest solution, which we do by minimising here.
```{prf:theorem} 
The maximal reachability probability is the unique solution to the following LP with variables $x_s$ for $s \in S$.
\begin{align*}
	\text{minimize}\quad & \sum_{s \in S} x_s & &   \\
	  & x_s = 1 & & \text{ for all $s \in T$} \\
	  & x_s = 0 & & \text{ for all $s \in \Smaxzero$} \\
	  & x_s \geq \sum_{s'} \delta(s,a)(s')x_{s'} & & \text{for all other $s$ and all $a\in \EnAct{s}$}
\end{align*}
```
````{prf:example}
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.lp import lp_maxreachprob
lp_maxreachprob(mdp_parker, "target")
```
````

### Consequences of the LP reduction
From the fact that linear programming can be solved in polynomial time, we get the following important consequence.
```{prf:theorem}
There is a polynomial-time algorithm to compute the value of an MDP.
```
In fact, the LP-based approach is the only known polynomial time algorithm for solving MDPs. 
We note, however, that practically, using LP solvers for MDPs is not very relevant and that most modern LP solvers (except Soplex and Z3) do not provide any formal guarantees about the quality of the solution. 


## Value Iteration

Intuitively, the idea of value iteration is to approximate the reachability probabilities by $n$-step reachability probabilities, for increasing $n$.
As such, the idea is similar to computing the set of reachable states by iteratively adding successor states in a breadth-first manner.
However, while in the qualitative case we reason about sets of states, now we must reason about the reachability probability per state.
More formally, we move from the [lattice](#def:lattice) $(2^S, \subseteq)$ where $2^S$ denotes the powerset on $S$ to the lattice $([0,1]^S, \preceq)$ where $\preceq$ denotes pointwise inequality.
```{note} Notation
Strictly, $2^S$ denotes the functions from $S$ to $\{ 0, 1 \}$, i.e., functions that determine membership for every state.
Likewise, $[0,1]^S$ denotes the functions from $S$ to $[0,1]$, i.e., functions that assign a probability to every state. Assuming a total order on states, $[0,1]^S$ can be interpreted as $|S|$-dimensional vectors, which is also convenient in examples.
```
To formally describe the iterative update of the reachability probabilities, we use _Bellman operators_. 
Where _Bellman equations_ describe the optimal solution, the Bellman operators are formal [operators](#app:fixpointoperators) that describe an update of values. 
We first present the operator for the minimal reachability probabilities, and then for maximal reachability probabilities.

### Minimal reachability probabilities
Recall the [Bellman equations](#thm:bellmaneq:minreachprob).
```{prf:definition} Bellman operator (MinReachProb)
:label: def:bellmanop:minreachprob
The Bellman operator for minimal reachability probabilities and for a fixed MDP is a mapping $\Phi \colon [0,1]^S \rightarrow [0,1]^S$ s.t. 
$$\Phi(F)(s) = \begin{cases} 
 	1 & \text{if }s \in T, \\
 	0 & \text{if }s \in \Szero, \\ 
\min_{a \in \EnAct{s}} \sum_{s' \in S} \delta(s,a,s') F(s') & \text{otherwise.}
 \end{cases}
$$
```
````{prf:example}
:label: ex:bellmanop:minreachparker

Recall the Bellman equations from @ex:bellman:minreachparker. We now show the Bellman operator:
```{code-cell} python
:tags: [remove-input]
equations = bellman.minreachprob(mdp_parker, "target", operator=True)
Math(r"\\".join([sympy.latex(eq) for eq in equations]))
```
Let us execute the Bellman operator, maybe first on the bottom element of the lattice, which assigns zero to every state.
```{code-cell} python
:tags: [remove-input]
operator = bellman.make_operator_minreachprob(mdp_parker, "target")
phione = operator.apply(bellman.zero_value(mdp_parker))
Math(r"\\".join(bellman.value_to_latex(phione, "\Phi(\mathbf{0})")))
```
Indeed, the probabilities we see here are the zero-step reachability probabilities to the target states.

If we apply the Bellman operator once more, we get the one-step minimal reachability probabilities:
```{code-cell} python
:tags: [remove-input]
phitwo = operator.apply(phione)
Math(r"\\".join(bellman.value_to_latex(phitwo, "\Phi(\Phi(\mathbf{0}))")))
```
And after one more application, the two-step reachability probabilities.
```{code-cell} python
:tags: [remove-input]
phithree = operator.apply(phitwo)
Math(r"\\".join(bellman.value_to_latex(phithree, "\Phi^3(\mathbf{0})")))
```
````

After looking at a concrete example, let us consider a more generic setting.
$\Phi(\mathbf{0})$ yields the indicator function for $T$, $\mathbb{1}_T$.
$\Phi(\mathbb{1}_T)$ yields the one-step minimal reachability probabilities to reach $T$. 
By simple substitution, we have $\Phi(\Phi(\mathbf{0})) = \Phi(\mathbb{1}_T)$.
Likewise, $\Phi(\Phi(\Phi(\mathbf{0})))$, denoted $\Phi^3(\mathbf{0})$ yields the two-step minimal reachability probabilities and indeed,
$\Phi^{n+1}(\mathbf{0})$ denotes the $n$-step minimal reachability probabilities.

```{prf:lemma} $n$-step interpretation
:label: lem:vi:nstep
$\Phi^n(\mathbf{0})(s)$ equals the minimum probability of reaching $T$ from $s$ within $n$ steps.
```
This $n$-step interpretation is the key intuition behind value iteration: each application of $\Phi$ extends the horizon by one step, and taking the limit recovers the unbounded reachability probability $V^{\min}(s)$.

```{prf:lemma}
The [Bellman operator for MinReachProb](#def:bellmanop:minreachprob) $\Phi$ is monotonic and $\omega$-continuous.
```

```{prf:theorem} MinReachProb is a fixpoint
Let $\Phi$ be the [Bellman operator for MinReachProb](#def:bellmanop:minreachprob). It holds that:
- The value is a unique fixpoint, i.e.: 
$$ \lfp{\Phi} = \gfp{\Phi} = V^{\min}, $$ and in particular also
$$\Phi(V^{\min}) = V^{\min}.$$
- $\Phi^n(\mathbf{0}) \leq V^{\min}$ for all $n$ and $\Phi^n(\mathbf{1}) \geq V^{\min}$ for all $n$.
```

#### Value iteration algorithm
The essence of value iteration is thus to start with $F \gets \mathbf{0}$ and repeatedly apply $F \gets \Phi(F)$ until some termination condition is met.
The problem is that we may not reach the fixpoint (the $n$-step reachability may always be smaller than the unbounded reachability).
What we realistically hope to achieve is $$|F - V^{*}| \leq \mathbf{\epsilon} \quad \text{(pointwise)}$$.

````{prf:example}
Consider the Bellman operator from @ex:bellmanop:minreachparker.
We run value iteration using this operator. 
```{code-cell} python
:tags: [remove-input]
vi = bellman.VI(operator, bellman.zero_value(mdp_parker))
results = [vi.step() for _ in range(8)]
bellman.visualise_iterations(results, background_gradient="viridis")
```
As the fixpoint is unique, we could alternatively start elsewhere and still converge against the optimum.
```{code-cell} python
:tags: [remove-input]
vi = bellman.VI(operator, bellman.one_value(mdp_parker))
results = [vi.step() for _ in range(8)]
bellman.visualise_iterations(results, background_gradient="viridis")
```
````

A natural stopping criterion for VI is to halt when successive iterates differ by at most $\varepsilon$ pointwise, i.e.\ $\|\Phi^{n+1}(\mathbf{0}) - \Phi^n(\mathbf{0})\|_\infty \leq \varepsilon$.
Unfortunately, this _local_ near-convergence does not imply that the current iterate is within $\varepsilon$ of $V^{\min}$.
The sequence $\Phi^n(\mathbf{0})$ does converge to $V^{\min}$ in the limit (the fixpoint is unique), but convergence can be arbitrarily slow for MDPs with end components: successive iterates may become indistinguishable while the current value is still far from $V^{\min}$ @DBLP:journals/tcs/HaddadM18.
Any finite iterate $\Phi^n(\mathbf{0})$ is a strict underestimate of $V^{\min}$, and stopping early yields no computable bound on the remaining error.

#### Interval iteration algorithm

Rather than relying on local near-convergence, _interval iteration_ (IVI) runs two VI sequences simultaneously: one from $\mathbf{0}$ (lower bound) and one from $\mathbf{1}$ (upper bound).
Because the fixpoint is unique, both sequences converge to $V^{\min}$, and the gap between them shrinks monotonically.
When lower and upper agree within tolerance $\varepsilon$, we have a guaranteed $\varepsilon$-approximation.

````{prf:example}
We run interval iteration on the Parker MDP for minimal reachability probabilities.
```{code-cell} python
:tags: [remove-input]
ivi = bellman.IVI(
    bellman.VI(operator, bellman.zero_value(mdp_parker)),
    bellman.VI(operator, bellman.one_value(mdp_parker)),
)
results = [ivi.step() for _ in range(8)]
bellman.visualise_iterations(results)
```
Each cell shows a `(lower, upper)` pair.
The bounds tighten with each step and converge to the same values as standard VI.
````



### Maximal reachability probabilities
```{prf:definition} Bellman operator (MaxReachProb)
:label: def:bellmanop:maxreachprob
The Bellman operator for maximal reachability probabilities (MaxReachProb) and for a fixed MDP is a mapping $\Phi \colon [0,1]^S \rightarrow [0,1]^S$ s.t. 
$$\Phi(F)(s) = \begin{cases} 
 	1 & \text{if }s \in T, \\
 	0 & \text{if }s \in \Smaxzero, \\ 
\max_{a \in \EnAct{s}} \sum_{s' \in S} \delta(s,a,s') F(s') & \text{otherwise.}
 \end{cases}
$$
```

The intuition behind the Bellman operator is the same as for the Bellman operator for minimal reachability probabilities.

```{prf:lemma}
The [Bellman operator for MaxReachProb](#def:bellmanop:maxreachprob) $\Phi$ is monotonic and $\omega$-continuous.
```
```{prf:theorem} MaxReachProb is a fixpoint
Let $\Phi$ be the [Bellman operator for MaxReachProb](#def:bellmanop:maxreachprob) for an MDP $\mdp$. It holds that:
- The value is the least fixpoint, i.e.: 
$$ \lfp{\Phi} = V^{\max}, $$ and in particular also
$$\Phi(V^{\max}) = V^{\max}.$$
- If $\mdp$ is [MEC-free](#def:mdp:mecfree), then there is a unique fixed point, i.e., $$ \lfp{\Phi} = V^{\max} = \gfp{\Phi}. $$
- $\Phi^n(\mathbf{0}) \leq V^{\max}$ for all $n$.
```

The application of (interval) value iteration is therefore similar to before. 
However, in contrast to the minimal reachability probability, it is important how we initialise value iteration when non-trivial MECs are present, as there can be multiple fixpoints.
The from-below sequence $\Phi^n(\mathbf{0})$ still provides a valid lower bound and converges to $V^{\max}$.
Meanwhile, the from-above sequence $\Phi^n(\mathbf{1})$ converges to the _greatest_ fixpoint, which strictly exceeds $V^{\max}$ (given non-trivial MECs).
Consequently, the gap between upper and lower bounds never closes, and interval iteration cannot certify convergence.
The standard remedy is to [eliminate maximal end components](#sec:eliminatemecs) first, which removes the spurious fixpoints and restores uniqueness.

````{prf:example}
We run interval iteration on the MDP from the MEC example (see @sec:mecs), computing the maximum probability of reaching $s_2$ (labeled $\mathit{target}$).
```{code-cell} python
:tags: [remove-input, remove-output]
mdp_mec = examples.create_mixed_mec_mdp()
op_mec = bellman.make_operator_maxreachprob(mdp_mec, "target")
ivi_mec = bellman.IVI(
    bellman.VI(op_mec, bellman.zero_value(mdp_mec)),
    bellman.VI(op_mec, bellman.one_value(mdp_mec)),
)
results_mec = [ivi_mec.step() for _ in range(8)]
```
```{code-cell} python
:tags: [remove-input]
bellman.visualise_iterations(results_mec)
```
The lower bounds for $s_3$ and $s_4$ converge to $0.7$ (the true value, reached via the escape action), but the upper bounds remain stuck at $1.0$.
They support each other: the Bellman update for $s_3$ sees $\max(\text{escape} \to 0.7,\; \text{loop} \to V(s_4) = 1)= 1$, and symmetrically for $s_4$.
The gap never closes.

After MEC collapsing (with the representative self-loop removed so the upper bound can descend), the equations have a unique fixed point and IVI converges:
```{code-cell} python
:tags: [remove-input, remove-output]
from stormvogel.stormpy_utils.mec import eliminate_mecs
mdp_col, _ = eliminate_mecs(mdp_mec, remove_representative_selfloops=True)
op_col = bellman.make_operator_maxreachprob(mdp_col, "target")
ivi_col = bellman.IVI(
    bellman.VI(op_col, bellman.zero_value(mdp_col)),
    bellman.VI(op_col, bellman.one_value(mdp_col)),
)
results_col = [ivi_col.step() for _ in range(6)]
```
```{code-cell} python
:tags: [remove-input]
bellman.visualise_iterations(results_col)
```
Both bounds meet by iteration 4.
````

```{admonition} Summary: interval iteration for min vs. max reachability
:class: tip

| Setting | Unique fixpoint? | From below $\Phi^n(\mathbf{0})$ | From above $\Phi^n(\mathbf{1})$ | Gap closes? |
|---------|:---:|---|---|:---:|
| Min, any MDP | Yes | Converges to $V^{\min}$ | Converges to $V^{\min}$ | Yes (possibly slowly) |
| Max, MEC-free | Yes | Converges to $V^{\max}$ | Converges to $V^{\max}$ | Yes |
| Max, with MECs | No | Converges to $V^{\max}$ | Converges to $\gfp{\Phi} > V^{\max}$ | No |

For min and MEC-free max, IVI terminates with a guaranteed $\varepsilon$-approximation.
For max with MECs, the upper bound stalls at the greatest fixpoint; MEC elimination is required before IVI can be applied.
```

## Dynamic programming
We briefly discuss a dynamic programming approach for [acyclic MDPs](#def:mdp:acyclic). Note that by definition, acyclic MDPs are also MEC-free. 
We therefore only discuss the minimal reachability probability case here; the maximal reachability probabilities can be computed exactly analogously. 
First, let us note that on acyclic models, VI terminates with the exact solution after $h$ steps, where $h$ is the length of the longest path to a sink state (without looping in the sink states).
This yields a quadratic run time for value iteration. By using dynamic programming, we can reduce the quadratic run time to a linear run time.

In particular, if we sort the MDP topologically (i.e., by ascending distance to the target), 
the following dynamic programming scheme yields the correct solution.

```{admonition} Algorithm
**Inputs** A Bellman operator and topologically sorted states $\mathsf{sorted}(S)$.

**Outputs** The least fixed point of that operator.

**for** every $s \in \mathsf{sorted}(S)$:

$\quad V(s) \gets \Phi(V)(s)$
```




```{prf:theorem}
Computing (minimal or maximal) reachability probabilities in acyclic MDPs can be done in linear time (in the size of the MDP).
```


## Combining methods
Beyond the three main families of algorithms, one can combine different algorithms.
The most relevant one is VI2PI: First VI, then PI. This yields the speed of VI (?!) and ensures an exact result with PI. 



# From reachability to temporal properties (and back)
So far, our exposition focussed exclusively on reachability probabilities. 
In particular, we categorised paths as good or bad, solely depending on whether these paths visit a particular state.
In this section, we support more general properties on paths.

(sec:dfa)=
## DFA properties
In a first step, we define paths that are accepted by a DFA.
More precisely, to support such properties, we label states with propositions.
```{prf:definition}
An MDP with states $S$ can be labelled by defining:
- a finite set of atomic propositions $\AP$,
- a labelling function $L\colon S \rightarrow 2^\AP$.
```
````{prf:example} 
We consider the following small MDP modelling a person navigating a small town.
States are labelled with $S$ (supermarket) or $L$ (library) when visiting those locations;
all other states carry no label.
```{code-cell} python
:tags: [remove-input]
from stormvogel.examples.minitown import create_minitown_mdp

mdp = create_minitown_mdp()
import stormvogel.to_dot            
sv.to_dot.plot_model_pydot(mdp)
```
````

````{prf:example} 
Consider the property _"visit both the supermarket and the library, in any order"_.
This is a DFA property over $\AP = \{S, L\}$.
The DFA has four states: $q_0$ (initial, neither visited), $q_1$ (supermarket visited first), $q_2$ (library visited first), and $q_3$ (accepting: both visited).
```{code-cell} python
:tags: [remove-input]
import stormvogel.dfa as dfa

aut = dfa.SymbolicDFA(
    states={"q0", "q1", "q2", "q3"},
    initial_state="q0",
    accepting_states={"q3"},
)
aut.add_transition("q0", lambda s: "L" in s and "S" not in s, "q2", label="¬S ∧ L")
aut.add_transition("q0", lambda s: "S" in s and "L" not in s, "q1", label="S ∧ ¬L")
aut.add_transition("q0", lambda s: "S" in s and "L" in s, "q3", label="S ∧ L")
aut.add_transition("q0", lambda s: "S" not in s and "L" not in s, "q0", label="¬S ∧ ¬L")
aut.add_transition("q1", lambda s: "L" in s, "q3", label="L")
aut.add_transition("q1", lambda s: "L" not in s, "q1", label="¬L")
aut.add_transition("q2", lambda s: "S" in s, "q3", label="S")
aut.add_transition("q2", lambda s: "S" not in s, "q2", label="¬S")
aut.add_transition("q3", lambda s: True, "q3", label="true")

dfa.plot_symbolic_dfa_pydot(aut)
```
````

To formalise what we want, we first lift paths to traces over these executions:

```{prf:definition} AP-trace
:label: def:mdp:aptrace
The $\AP$-trace of a path $\xi = s_0a_0s_1a_1 \dots$ omits the actions and lifts states to the labels: $$ \aptrace{\xi} = L(s_0)L(s_1) \dots \in \big({2^\AP}\big)^{*}$$.
The set of all $\AP$-traces is called $\ApTrace$.
```

In a Markov chain, we can easily lift the probability of a path to the probability of a trace:
$$ \pr(\tau) = \sum_{\xi \in \Paths, \aptrace{\xi} = \tau} \pr(\xi). $$
Given a set of traces $X \subseteq \ApTrace$, we can thus also define $\pr(X)$ as the probability of generating 
a path $\xi$ s.t. $\aptrace{\xi} \in X$.

We consider properties that can be described by [deterministic finite automata](#app:dfa).
Given an MDP with atomic propositions $\AP$, a DFA with alphabet $2^{\AP}$ describes a _DFA_ property.
```{prf:definition}
   Given a Markov chain with atomic propositions $\AP$ and a DFA $\dfa$ with alphabet $2^{\AP}$:
   Let $\mathcal{L} \subseteq \big({2^\AP}\big)^{*}$ be the [language](#def:dfa:language) of $\dfa$.
   The satisfaction property for the DFA property $\pr(s \models \dfa)$ is given as:
   $$\pr(s \models \dfa) = \pr(\{ \xi \in \Paths(s) \mid \aptrace{\xi} \in \mathcal{L} \}).$$
```
For MDPs, we then lift to minimal and maximal probabilities analogously to reachability probabilities.
Note however, that memoryless policies are not sufficient.
Intuitively, satisfying a DFA property may require counting steps or remembering which labels have been seen, which a memoryless policy cannot do.
For instance, a step-bounded property requires knowing how many steps have been taken, and visiting both the supermarket and the library requires remembering which one was visited first.
```{prf:lemma}
:label: lem:dfa:memoryless-insufficient
There exist MDPs and DFAs such that:
1. $$\sup_{\pi \in \Policies} \pr^\pi(s \models \dfa) > \max_{\pi \in \MdPolicies} \pr^\pi(s \models \dfa) $$
2. $$\inf_{\pi \in \Policies} \pr^\pi(s \models \dfa) < \min_{\pi \in \MdPolicies} \pr^\pi(s \models \dfa) $$
```

### Reduction to reachability
We can compute the (minimal/maximal) probability to satisfy a DFA property by a product construction.
```{prf:definition} 
Let $\mdp = \langle S, A, \delta_\mdp$ be an MDP with initial state $\sinit$, atomic propositions $\AP$ and labelling function $L$,
 and let $\dfa = \langle Q, \AP, q_0, \delta_\dfa, \Acc \rangle$ be a DFA. 
We define the product MDP $$\mdp \otimes \dfa = \langle S \times Q, A, \delta' \rangle$$ 
with initial state $\langle \sinit, \delta_\dfa(q_0, L(\sinit)) \rangle$ 
and a transition relation such that
$$\delta'(\langle s, q \rangle, a)(\langle s', q' \rangle) = \delta_\mdp(s,a)(s') \cdot \indicator{\delta_\dfa(q,L(s')) = q'}$$.
```
Specifically, the state space contains the MDP state and the DFA state. In every transition, we update the DFA state by reading the label of the _new_ MDP state $s'$.
The initial DFA state is also determined by reading the label of the initial MDP state immediately, which is why the initial product state is $\langle \sinit, \delta_\dfa(q_0, L(\sinit)) \rangle$ rather than $\langle \sinit, q_0 \rangle$.

```{prf:remark}
The DFA must be _complete_: every DFA state must have exactly one outgoing transition for every possible label $\sigma \in 2^\AP$.
This ensures that the product transition $\delta'$ is well-defined for every MDP transition.
```

In the product, we can then define target states as states that correspond to an accepting state in the DFA.
```{prf:theorem}
For any MDP $\mdp$ with states $S$, DFA property $\dfa$ with accepting states $\Acc$, and $\opt \in \{ \min, \max \}$
$$ \pr^{\opt}_\mdp(s \models \dfa) = \pr^{\opt}_{\mdp \otimes \dfa}(\lozenge \{ \langle s,q \rangle \mid q \in \Acc  \}). $$
```
```{prf:lemma}
For any MDP $\mdp$ and DFA $\dfa$, the optimal reachability probability on $\mdp \otimes \dfa$ is achieved by a memoryless deterministic policy.
Such a policy uses the DFA state as memory and therefore corresponds to a [finite-state controller](#def:fsc) on the original MDP $\mdp$ with memory states $Q_\dfa$.
In particular, the supremum over all policies in $\mdp$ is attained and equals the maximum over FSCs with memory states $Q_\dfa$.
```
This is why the product construction resolves the insufficiency of memoryless policies established in @lem:dfa:memoryless-insufficient.

````{prf:example} 
The following visualises the product MDP for the DFA and the MDP above. 
```{code-cell} python
:tags: [remove-input]
import stormvogel.to_dot            
sv.to_dot.plot_model_pydot(dfa.product(mdp, aut))
```
````

### Reach-avoid and step-bounded properties
We highlight two simple instances of DFA properties. 
Algorithmically, one typically avoids building the full product for these properties, but semantically, it is adequate to think about the product.

#### Reach-avoid properties
In a _reach-avoid_ property, one aims to reach the target states $T$ while avoiding another set of bad states $B$. 
For simplicity, let us assume that reach and avoid states are disjoint.
A reach-avoid property can be simply captured by a 3-state DFA: 
- The initial state,
- an absorbing, accepting state reached after seeing a target in the MDP,
- an absorbing, not accepting state, reached after seeing an avoid state in the MDP.
In particular, exactly the paths that visit a reach state, before reaching an avoid state will be accepted by the DFA.
Below, we exemplify the construction. For the sake of completeness, we assume that a path that visits target states and bad states simultaneously should not be accepted.
````{prf:example}
```{code-cell} python
:tags: [remove-input]
import stormvogel.dfa as dfa

aut = dfa.SymbolicDFA(
    states={"q0", "q_acc", "q_rej"},
    initial_state="q0",
    accepting_states={"q_acc"},
)

# From the initial state:
aut.add_transition("q0", lambda s: "T" in s and "B" not in s, "q_acc", label="T ∧ ¬B")
aut.add_transition("q0", lambda s: "B" in s, "q_rej", label="B")
aut.add_transition("q0", lambda s: "T" not in s and "B" not in s, "q0", label="¬T ∧ ¬B")

# Absorbing states
aut.add_transition("q_acc", lambda s: True, "q_acc", label="true")
aut.add_transition("q_rej", lambda s: True, "q_rej", label="true")

dfa.plot_symbolic_dfa_pydot(aut)
```
````

#### Step-bounded properties
A _step-bounded_ property with horizon $h$ asks for reaching a target state within $h$ steps.
A DFA for this property consists of $h+2$ states: 
We start in a state encoding that has made zero steps so far. 
If a target state is visited in the MDP, we go to an absorbing and accepting state. 
Otherwise, we go from the state encoding $i$ steps so far to $i+1$ steps so far.
The state encoding $h+1$ steps is absorbing and should be interpreted as having exceeded the horizon.
Step-bounded properties can be generalised to [cost-bounded properties](#sec:costbounded).


## Büchi properties
Reachability and DFA properties consider paths up to some (indefinite) horizon. 
Instead, in Büchi properties, we consider infinite paths.
Essentially, Büchi properties ask that a set of (target) states are visited infinitely often along a path. 
```{prf:remark}
To avoid confusion, we call the _target_ states _Büchi_ states in this subsection.
```
```{prf:definition}
Given a set of _Büchi_ states $T$, an infinite path $s_0s_1\dots$ satisfies a Büchi property $\Box\lozenge T$ iff 
there are infinitely many $i$ such that $s_i \in T$.
```
As such, Büchi properties allow expressing that one can reach some good state and that one is able to visit from that state another good state.
They help to state that there is not a bounded number of recharge actions, etc.
Büchi properties are a specific type of so-called limit properties, where the acceptance of a path is really determined by whatever happens infinitely often. 
In finite systems, this specifically means that we can concentrate our analysis on end-components.
```{prf:theorem}
For any $\pi$:
$$ \pr^\pi( \xi \in \Paths \mid \!\!\!\underbrace{\infinite{\xi}}_{\text{choices visited inf often}}\!\!\! \text{ is an EC } ) = 1	$$
```
The proof first observes that for any infinite path in a finite MDP, there must be at least one choice which is visited infinitely often.
Every successor state of that choice must also be visited infinitely often, and thus in those states, also a choice is visited infinitely often. 
This ascending set then stabilises with a strongly connected component where a policy can stay forever, i.e., an EC.

```{prf:definition}
An EC satisfies a Büchi property $\varphi$ if it contains a target state. 
```
We justify this wording by remarking that a policy can choose to visit any state in an EC infinitely often. In particular, such a policy can thus satisfy $\varphi$.

```{prf:theorem} 
- Let $U_\varphi$ be the union of all $S'$ in an EC that satisfies the Büchi property $\varphi$. 	
- Let $V_\varphi$ be the union of all $S'$ in an EC that does not satisfy the Büchi property $\varphi$.
Then:
1. $\pr^{\max}(\varphi) = \pr^{\max}(\lozenge U_\varphi)$
2. $\pr^{\min}(\varphi) = 1 - \pr^{\max}(\lozenge V_\varphi)$.	
```
The two parts of the theorem have a pleasing duality.
For the maximum, a policy satisfies $\varphi$ if and only if it eventually reaches an EC with a target state; so maximising $\pr(\varphi)$ is the same as maximising the probability of reaching $U_\varphi$.
For the minimum, the adversary (trying to minimise $\pr(\varphi)$) wants to reach an EC without a target state and stay there; so minimising $\pr(\varphi)$ is equivalent to maximising the probability of reaching $V_\varphi$, and thus $\pr^{\min}(\varphi) = 1 - \pr^{\max}(\lozenge V_\varphi)$.

The theorem above gives rise to a (naive) algorithm: Compute all ECs, classify each EC as satisfying or not satisfying, and then solve a reachability probability problem. 
Note that such an algorithm is not efficient, as there are exponentially many ECs. 
However, careful graph algorithms with a nested fixpoint operation avoid the necessity to precompute all ECs, while in spirit identifying ECs.

````{prf:example}
We illustrate the reduction on a 6-state MDP with Büchi state $s_1$.
```{code-cell} python
:tags: [remove-input]
import stormvogel.model as sv_model
import stormvogel.bird as bird
import stormvogel.to_dot as to_dot

def create_buchi_mdp():
    def _available_actions(s):
        if s in [0, 1, 2, 4]: return ["a", "b"]
        return ["a"]  # s3, s5 have only one action

    def _delta(s, act):
        if s == 0:
            return [(0.7, 1), (0.3, 4)] if act == "a" else [(0.4, 1), (0.6, 4)]
        if s == 1:
            return [(1.0, 2)] if act == "a" else [(1.0, 1)]   # a: s1→s2, b: self-loop
        if s == 2:
            return [(1.0, 3)] if act == "a" else [(1.0, 2)]   # a: s2→s3, b: self-loop
        if s == 3:
            return [(1.0, 1)]                                  # a: s3→s1
        if s == 4:
            return [(1.0, 5)] if act == "a" else [(1.0, 4)]   # a: s4→s5, b: self-loop
        if s == 5:
            return [(1.0, 4)]                                  # a: s5→s4

    def _labels(s):
        if s == 1: return ["T"]
        return []

    def _friendly_name(s):
        return ["s0", "s1", "s2", "s3", "s4", "s5"][s]

    return bird.build_bird(_delta, available_actions=_available_actions, init=0,
        labels=_labels, modeltype=sv_model.ModelType.MDP, friendly_names=_friendly_name)

mdp_buchi = create_buchi_mdp()
to_dot.plot_model_pydot(mdp_buchi)
```

We enumerate all ECs, classify each by whether it contains the Büchi state, and label every EC state as $U$ or $V$:

```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.mec import enumerate_ecs, detect_mecs
from stormvogel.stormpy_utils.model_checking import model_checking

target_states = mdp_buchi.get_states_with_label("T")
mec_state_sets = {frozenset(mec) for mec in detect_mecs(mdp_buchi)}

# one representative per unique state set; label all EC states U or V
seen = {}
for ec in enumerate_ecs(mdp_buchi):
    label = "U" if ec.satisfies_buchi(target_states) else "V"
    for s in ec.states:
        s.add_label(label)
    if ec.states not in seen:
        seen[ec.states] = (label, ec)

import pandas as pd
rows = [
    (
        "{" + ", ".join(sorted(s.friendly_name for s in states)) + "}",
        "✓" if states in mec_state_sets else "✗",
        "✓" if label == "U" else "✗",
    )
    for states, (label, ec) in seen.items()
]
pd.DataFrame(rows, columns=["States", "MEC", "Sat. φ"])
```

Notice that $s_2$ appears in both a satisfying EC ($\{s_1, s_2, s_3\}$) and the non-satisfying EC $\{s_2\}$ (✗ in the last column).
Using only MECs would miss the latter, leaving $s_2 \notin V_\varphi$ and giving the wrong answer for $\Pr^{\min}$.

```{code-cell} python
:tags: [remove-input, remove-output]
res_max   = model_checking(mdp_buchi, 'Pmax=? [F "U"]')
res_min_v = model_checking(mdp_buchi, 'Pmax=? [F "V"]')

s0 = mdp_buchi.initial_state
pr_max     = res_max.at(s0)
pr_min_v   = res_min_v.at(s0)
pr_min     = 1 - pr_min_v
```

Thus $\Pr^{\max}(\varphi) = \Pr^{\max}(\lozenge U_\varphi) =$ {eval}`f"{pr_max:.4f}"` and
$\Pr^{\min}(\varphi) = 1 - \Pr^{\max}(\lozenge V_\varphi) = 1 -$ {eval}`f"{pr_min_v:.4f}"` $=$ {eval}`f"{pr_min:.4f}"`.
````

```{attention}
Skipped for now.
```

## Towards LTL and $\omega$-regular
```{attention}
Skipped for now.
```

# Rewards
We now extend MDPs with rewards. Rewards are commonly used to describe resource usage (time, energy, ...). 
They can also be used to define desired and undesired actions or states.

```{prf:definition} MDP with rewards
An MDP with states $S$ may have 
 - _state rewards_ $r \colon S \rightarrow \mathbb{Q}_{\geq 0}$[^foot:mixing] or,
 - _choice rewards_ $r \colon S \times A \rightarrow \mathbb{Q}_{\geq 0}$.
 - _transition rewards_ $r \colon S \times A \times S \rightarrow \mathbb{Q}_{\geq 0}$.
```
[^foot:mixing]: Mixing positive/negative rewards requires more cornercases and is therefore not part of this material.

We consider choice rewards in the lecture notes (which coincides with state rewards obtained upon exiting a state in a Markov chain).
```{note} Cost vs reward.
Rewards and costs are used mostly interchangeably. 
As a general rule of thumb, the word reward is preferred when one tries to maximise (or bound from below) the quantity, 
whereas cost is preferred when one tries to minimise (or bound from above) the quantity.
In these lecture notes, we stick to rewards.
```

```{prf:definition} Reward of a path
:label: def:rewardpath
Given a path $\xi = s_0a_0s_1\dots$, the reward of $\xi$ is defined as $$\mathsf{rew}(\xi) = \sum_i = r(s_i, a_i)$$. 
```


## Expected reachability rewards
So far, we have partitioned infinite (or finite) paths into good and bad paths. 
We've then computed policies that optimise the probability mass of good paths.
In this section, we use the annotation with rewards to weigh every path and optimise the _expected_ reward, 
i.e., the weighted sum of the probability and the weight of every path.  

Consider the reachability reward for infinite paths:
$$ 
\mathsf{rew}_{\lozenge T}(s_0s_1s_2 \dots) = \begin{cases}
 \mathsf{rew}(s_0a_0 \dots s_n) & \text{if } s_i \not\in T \text{ for all }i < n \text{ and } s_n \in T \\
 \infty & \text{otherwise.}	
 \end{cases} 
$$

### Expected reachability rewards in Markov chains.
```{prf:definition} 
In Markov chains, the expected reward until reaching $T$, $\mathbb{E}(\lozenge T)$ is defined as:
$$  
	\sum_{r=0}^\infty r \cdot \pr(\{\xi \in \lozenge T \mid \mathsf{rew}_{\lozenge T}(\xi) = r \}).
$$
```
```{prf:lemma}
The expected reward is finite iff the the target states are reached with probability one. 
```

For expected rewards, a standard linear equation system can be formulated:
```{prf:theorem}
Given a Markov chain with target states $T$. Assume the expected reward until reaching $T$ is finite. 
 Then the expected reward until reaching $T$ is the unique solution to the linear equation system over real-valued variables $x_s$, $s \in S$:
$$
x_s = \begin{cases} 0 & \text{ if }s \in T \\
       r(s) + \sum_{} P(s)(s') \cdot x_{s'} & \text{ otherwise}
       \end{cases}
$$
```

### Maximal expected reachability rewards
For MDPs, we can study reachability rewards under a minimising or a maximising policy. We write $\mathbb{E}^{\pi}(\lozenge T)$ for the expected reward in the Markov chain induced by $\pi$.
We start with maximal expected rewards. As before, we definite the maximal expected reward as
$$\mathbb{E}^{\max}(\lozenge T)  = \sup_{\pi \in \Policies} \mathbb{E}^{\pi}(\lozenge T). $$
Below, we follow the exposure in @DBLP:conf/tacas/ChatterjeeQSWWZ25 [Section 5].
In particular, we define another Bellman operator.
```{prf:definition}
Let $\mathcal{\Phi}_{\mathbb{E}}^{\max}\colon \mathbb{R}_{\infty,\ge 0}^{S}\to \mathbb{R}_{\infty,\ge 0}^{S}$ 
$$
\mathcal{\Phi}_{\mathbb{E}}^{\max}(F)(s) =
\begin{cases}
0 & \text{if } s\in T,\\
\max_{a\in \EnAct{s}} \operatorname{rew}(s,a) + \sum_{s'}\delta(s,a,s') \cdot F(s')
& \text{if } s \in \Sposmin,\\
\infty & \text{otherwise}
\end{cases}
$$
```
In the operator, we see that we explicitly assign infinite reward whenever there is a possibility for a policy to reach the target with probability zero.
```{prf:theorem}
$$\lfp{\mathcal{\Phi}_{\mathbb{E}}^{\max}} =  \mathbb{E}^{\max}(\lozenge T). $$
```
It also follows that there is a memoryless deterministic policy that optimises the expected reward. 
We can compute the expected reward by value iteration or policy iteration.

### Minimal expected reachability rewards
```{attention}
Skipped for now.
```

### Reachability probabilities as expected rewards
Expected reachability rewards can express reachability probabilities. 

## Expected discounted total rewards
### Discounted rewards in Markov chains
In a lot of MDP literature on planning (or control) problems, infinite horizon (total rewards) are studied. 
That is, there are no target states and the reward along a path may diverge. 
A common mathematical trick is to exponentially discount reward with a factor $\gamma \in (0,1)$.

The discounted reward of a (possibly infinite) path is then defined as:
$$ \drew_\gamma(s_0a_0s_1a_1 \dots) = \sum_i \gamma^i r(s_i) $$.
We note that this sum always exists (and is finite) as $r(s)$ is bounded from above.

The discounted expected reward for a Markov chain is then defined as:
```{prf:definition} Discounted reward
For Markov chain $\mdp$ and $\gamma \in (0,1)$, we define the expected discounted reward $\mathbb{E}_\mdp(s \models \mathsf{tot}_\gamma)$  as 
$$\mathbb{E}_\mdp(s \models \mathsf{tot}_\gamma) = \int_{\xi \in \Paths(s)} \pr(\xi) \cdot \drew_\gamma(\xi).$$
```
As $\drew_\gamma$ is bounded from above, the integral exists.

Discounting the future is well-motivated in finance, where immediate gains are worth more than later gains.
It is also commonly motivated to be useful to prioritize nearby good actions over future actions (as a model may not predict the long-term future well).	
In general, it is fair to say that with discounted rewards, one may find useful policies for a planning problem (where the goal is to find the policy).
However, the correct interpretation of a discounted reward or any guarantee about it is hard and subject to modelling decisions.

```{attention}
Content missing: Express total discounted reward as reachbility reward
```

### Optimal discounted total rewards in MDPs
We can now define $\mathbb{E}^{\max}(s \models \mathsf{tot}_\gamma)$ and $\mathbb{E}^{\min}(s \models \mathsf{tot}_\gamma)$
analogously to $\pr^{\min}$ and $\pr^{\max}$. As before, memoryless deterministic policies suffice.

#### Bellman operator
Likewise, we can define Bellman equations and the operators. For conciseness, we only define the operator:
```{prf:definition} Bellman operator (MaxDiscReward/MinDiscReward)
:label: def:bellmanop:mdiscountedexpreward
The Bellman operator for maximal discounted total reward and for a fixed MDP is a mapping $\Phi_{\drew_\gamma}^{\max} \colon [0,\infty)^S \rightarrow [0,\infty)^S$ s.t. 
$$\Phi_{\drew_\gamma}^{\max}(F)(s) = 
\max_{a \in \EnAct{s}} r(s,a) + \gamma \cdot \sum_{s' \in S} \delta(s,a,s') F(s').
$$
The Bellman operator for minimal discounted total reward is defined analogously. 
```
We observe that this Bellman operator is a contraction operator and that thus, Banach's fixpoint theorem applies.
In particular:
```{prf:theorem}
For the Bellman operators for discounted total reward the following holds:
$$\lfp{\Phi_{\drew_\gamma}^{\max}} = \gfp{\Phi_{\drew_\gamma}^{\max}} =  \mathbb{E}^{\max}(\mathsf{tot}_\gamma) $$
and
$$ \lfp{\Phi_{\drew_\gamma}^{\min}} = \gfp{\Phi_{\drew_\gamma}^{\min}} =  \mathbb{E}^{\min}(\mathsf{tot}_\gamma). $$ 
```
#### Algorithms
PI, VI, and LP all apply.
Furthermore, PI and VI can compute $\varepsilon$-precise results in polynomial time. 
While using a discount factor towards one converges against the undiscounted total reward, 
the performance of VI and PI quickly degrades with higher discount factors.

```{prf:remark}
The undiscounted total reward can also be defined,
but will be infinite if there exists a reachable MEC with non-zero reward.
If no such MEC exists, then this property can be rewritten
as reachability reward in a modified MDP with a fresh target state.
See also 
```


(sec:costbounded)=
## Cost-bounded reachability 
```{attention}
All content here is still missing.
```
