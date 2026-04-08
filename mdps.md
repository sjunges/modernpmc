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

# Markov Decision Processes
```{attention}
With the attention blocks, we highlight blocks that are still missing. 
Additionally, various examples are still missing and most citations are still missing
```

```{code-cell} python
import stormpy
import sympy
import stormvogel as sv
import stormvogel.teaching as teach
import stormvogel.bird as bird
sympy.init_printing()
from IPython.display import Math
import stormvogel.teaching.bellman as bellman
```


```{code-cell} python
:tags: [remove-input, remove-output]

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
        return "s"+str(s)
        
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
        return "s"+str(s)
        
    def _friendly_name(s):
        return "s"+str(s)
    
    return bird.build_bird(
        _delta, available_actions=_available_actions, init=0, labels=_labels, modeltype=sv.ModelType.MDP, friendly_names=_friendly_name
    )
        
mdp_parker = create_mdp_parker()
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
Later we extend MDPs with _atomic propositions_, _target states_, and _reward functions_.

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

For any choice $\langle s,a\rangle$ we may write $\delta(s,a,s')$ to denote $\delta(s,a)(s')$.

```{warning} Assumptions
Unless stated otherwise we assume

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

The set of all paths is denoted $\Paths^\mdp$.
```
For a finite path
$
\xi = s_0 a_0 s_1 \dots s_n
$ 
we write $\last{\xi}$ for the final state $s_n$.

For an infinite path
$
\xi = s_0 a_0 s_1 a_1 s_2 \dots
$
we define $$ \infinite{\xi} = \{ \langle s, a \rangle  \mid |\{ i \mid \langle s_i, a_i \rangle = \langle s, a \rangle \}| = \infty \} $$
as the choices that are made infinitely often along a path.

### Graph structure
(def:mdp:sinkstate)=
A _sink state_ is a state $s$ with $\delta(s,a)(s) = 1$ for all $a \in \EnAct{s}$.

(def:mdp:acyclic)= 
An MDP is _acyclic_ if along every infinite path, the only states visited infinitely often are sink states.

### Policies
Policies resolve nondeterminism
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

We denote the set of all policies with $\Policies$ and use $\MdPolicies$ for the memoryless deterministic policies

Memoryless deterministic policies can be written as
$
\pi \colon S \rightarrow A.
$

We furthermore define the set of paths under a policy $\Paths^\pi$ as the set of policies that is consistent with a policy. 


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
It is useful to consider a special case for memoryless (possibly randomised) policies
`````{prf:definition} Induced Markov chain (memoryless policies)
For a memoryless policy $\pi$, the induced Markov chain is $\mdp[\pi] =
\langle S , \delta^\pi \rangle$
where
```{math}
:enumerated: false
\delta^\pi(s)(s') = \sum_a \pi(s)(a)\delta(s,a)(s'). 
```
`````

```{prf:remark} Two definitions?
These two constructions yield equivalent Markov chains
(formally: almost bisimilar).
```

# Reachability probabilities

For a policy $\pi$, the _reachability probability_
$
\pr^\pi_\mdp(s \models \lozenge T)
$
denotes the probability of eventually reaching a target state $T$ from a state $s$. 
The probability $\pr^\pi_\mdp(s \models \lozenge T)$ is defined via the induced MC $\mdp[\pi]$.
We are mostly interested in this probability from the initial state $\sinit$, in which case we simplify notation 
$\pr^\pi_\mdp(\lozenge T)$.

```{admonition} Problem: Standard verification problem for reachability probabilities
Given MDP $\mdp$, a _threshold_ $\lambda \in \mathbb{Q}$  and $\bowtie \in \{\leq,\geq\}$, decide whether

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

## Qualitative verification

Qualitative verification refers to the setting where the threshold $\lambda$ is either zero or one.
As we will see, qualitative verification can be solved without reference to the precise probabilities in the MDP.
Assume fixed $\mdp$ and target set $T$.

### Possible reachability (max)
One of the simplest questions we can ask is to find the set of states that can reach the targets with positive probability.

More formally, we denote this set of states  $$\Spos = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) > 0 \}$$, i.e., the set of states where the _maximum_ probability over all policies is positive. 
It holds that 

$\Spos = S \setminus \Smaxzero$
using
```{math}
:enumerated: false
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


```{attention}
Discussion that graph reachability is a least fixed point is missing.
```

### Possible reachability (min)
We now study computing
$$ \Sposmin = \{ \forall \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) > 0 \}, $$
that is, the complement of $\Szero$. Concretely, we will use that  each policy reaches the target with positive probability iff it has a path to the target of length $n$.
This can be computed as a least fixpoint. Specifically, we define 

$$
\Psi_{{\min}>0}\colon 2^S \rightarrow 2^S
$$
such that
$$
\Psi_{{\min}>0}(X)=
\{ s \in S \mid \text{for all } a \in \EnAct{s} \exists s' \in T_i. \delta(s,a)(s') > 0  \} \cup T
$$
The operator is monotonic, the lattice is finite. 

```{prf:theorem}
$\lfp{\Psi_{{\min}>0}} = \Sposmin$.
```


### Almost-sure reachability (max)
We are also interested in computing the set of states from which it is possible to ensure that we almost-surely reach the target states.

More formally, we denote this set of state $$\Smaxas = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 0 \}.$$

We use a recursive equation to characterize this set. If $s \in T$, then clearly $s \in \Smaxas$. Otherwise, for $s \not \in T$:  $$s \in \Smaxas \text{\quad iff \quad} \exists a \in \EnAct{s}. \forall s' \in \supp{\delta(s,a)}. s' \in \Smaxas. $$

To compute a the set $\Smaxas$, we want to provide an iterative procedure, which operates on sets of states.
Specifically, we define the operator

$$
\Psi_{{\max}=1}\colon 2^S \rightarrow 2^S
$$
such that
$$
\Psi_{{\max}=1}(X)=
\{ s \mid
\exists a\in\EnAct{s}.
\forall s'\in\supp{\delta(s,a)}.
s'\in X \}\cup T
$$
The operator is monotonic, the lattice is finite. 

```{prf:theorem}
$\gfp{\Psi_{{\max}=1}} = \Smaxas$.
```

```{attention}
The explicit algorithm discussed in the lecture is still missing.
```
### Almost-sure reachability (min)
Finally, we can also define the set of states $$\Sminas = \{ s \mid \forall \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 0 \}.$$


```{attention}
Not discussed in the lecture. 
```

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
\min_{a\in A(s)}
\sum_{s'}\delta(s,a,s')x_{s'}
& \text{otherwise}
\end{cases}
$$
These equations are called the Bellman equations _for minimal reachability probabilities_.
```

```{code-cell} python
:label: fig:mdpparkervis
:caption: Visualisation of an MDP to illustrate the [Bellman equations for MinReachProb](#thm:bellmaneq:minreachprob). 
:tags: [remove-input]

vis = sv.show(mdp_parker)
```

````{prf:example}
:label: ex:bellman:minreachparker
Consider the MDP from @fig:mdpparkervis. The Bellman equations for minimal reachability probability are:
```{code-cell} python
:tags: [remove-input]
equations = bellman.minreachprob(mdp_parker, "s2")
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
2. There exist a policy $\pi^{*}$ such that for all $s \in S$:
$$ \pr^{\pi^{*}}(s \models \lozenge T) = \min_{\pi \in \MdPolicies} \pr^\pi(s \models \lozenge T).$$
```
For any solution to the Bellman equations, it simple to extract **a** witnessing memoryless deterministic policy $\pi^{*}$---one simply takes $$\pi^{*}(s) = \argmin_{a} \sum_{s'} \delta(s,a)(s') x_{s'}.$$
We highlight that there is a unique solution to the Bellman equation, but not a unique minimizing policy. 
Furthermore, the theorem justifies talking about minimising policies for a target set, but independently of a specific state.



### Maximal reachability probability
While most aspects of computing minimal and maximal reachability probabilities are analogously, the theorem about the Bellman equations differ for both cases.
```{prf:theorem} Bellman equations (MaxReachProb)
:label: thm:bellmaneq:maxreachprob
Given an MDP with states $S$. Consider variables $x_s$ for each $s \in S$.
The maximal reachability probability equals the _minimal_ solution

$$
x_s=
\begin{cases}
1 & s\in T \\
0 & s\in \Smaxzero \\
\max_{a\in A(s)}
\sum_{s'}\delta(s,a,s')x_{s'}
& \text{otherwise}
\end{cases}
$$
```

```{code-cell} python
:label: fig:mdponevis
:caption: Visualisation of an MDP with no unique solution for [Bellman equations for MaxReachProb](#thm:bellmaneq:maxreachprob). 
:tags: [remove-input]

vis = sv.show(mdp_one)
```

````{prf:example} 
:label: ex:maxreachprobnotunique
Consider the MDP in @fig:mdponevis. The [Bellman equations for MaxReachProb](#thm:bellmaneq:maxreachprob) are:

```{code-cell} python
:label:eq1
:tags: [remove-input]
equations = bellman.maxreachprob(mdp_one, "s1")
Math(r"\\".join([sympy.latex(eq) for eq in equations]))
```
In particular, any assignment to $x_0 \geq 0.5$ is part of a valid solution. However, @thm:bellmaneq:maxreachprob clarifies that only $x=0.5$ is a valid solution. 
````
In general, the problem we observe here is that there are (sets of) states where a policy can choose to stay indefinitely. 
If these states do not include target states, then staying there forever means that the states have reachability probability $0$,
yet the Bellman equations do not enforce that these states must be assigned to zero. 
In @sec:mecs, these sets of states are called end-components and we show that (roughly) in MDPs without such end-components, the [solutions to the Bellman equations are unique](#thm:mdp:mecfreemaxreach).


```{prf:theorem} Memoryless policies suffice (MaxReachProb)
:label: thm:mdsuffices:maxreachprob
For any MDP and target set $T$:
1. For any state $s$ it holds that
$$\sup_{\pi \in \Policies} \pr^\pi(\lozenge T) = \max_{\pi \in \MdPolicies} \pr^\pi(\lozenge T) $$
2. There exist a policy $\pi^{*}$ such that for all $s \in S$:
$$ \pr^{\pi^{*}}(s \models \lozenge T) = \min_{\pi \in \MdPolicies} \pr^\pi(s \models \lozenge T).$$
```
As there is no unique solution to the Bellman equations when maximising, extracting an optimal policy is harder. 
In particular, while any optimal policy $\pi$ must satisfy $$\pi(s) = \argmax_{a} \sum_{s'} \delta(s,a)(s') x_{s'},$$ this condition is not sufficient. 
```{prf:example}
Consider @ex:maxreachprobnotunique. Both actions in $s_0$ will yield value $0.5$.
```
Instead, we must find a policy that makes progress towards to the targets.
```{attention}
Content missing.
```


### The notion of value
To simplify the exposition, it is customary to merge discussions about minimal and maximal reachability probabilities to a fixed set of target states $T$.
If we know from context we want to compute the  (minimal or maximal) reachability probability  from state $s$ for a set of target states $T$, we can  call
- the probability induced by a policy _the value of the policy_ (from state $s$),
- the minimal (resp maximal) probability  from a state $s$ _the value of $s$_,
- the value of the initial state is the _value of the MDP_.
- We use $V^\pi_\mdp(s)$ to define the value induced by a policy and  $V^{*}_\mdp(s)$ to define the value of a state. We omit $\mdp$ whenever possible.
```{danger} Implicit notation
The use of values of MDPs etc is always contextual, and this is never clear from the notation. We sometimes write $V^{min}$ or $V^{max}$ instead of $V^{*}$ to be more explicit.
```
```{note} Value beyond reachability probabilities
Later, we will also use value for other properties, including (discounted or total) expected rewards, and the probabilities for temporal properties.
```

(sec:mecs)=
## Maximal end components
End-components are sub-MDPs -- induced by a set of choices (state-actions pairs) where a policy can visit every state and stay forever.
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
$$ \pr_s^\pi(\{ \xi \in \Paths^\pi(s) \mid \infinite(\xi) \text{ is an EC}   \}) = 1 $$
```
That is, the probability that the choices visited infinitely often from any state onwards form an EC is one. 
In particular, that means that there are still paths that do not eventually end up in an EC, as there are paths that take any loop infinitely often.

End-components can overlap and can contain other end-components. 
It is often helpful to consider __maximal end components__ (MECs): 
Maximal end components are end components not contained in any end component.
MECs cannot overlap. 
MECs can be detected with an efficient graph algorithm @BK08 [Algorithm 47]. 

(def:mdp:trivialmec)=
We note that sink-states with their self-loops form are always MECs: We call these _trivial_ MECs.

(def:mdp:mecfree)= 
Analogously to the notion of an [acyclic MDP](#def:mdp:acyclic),
we call an MDP _MEC-free_, if all MECs are [trivial](#def:mdp:trivialmec).

```{note}
Importantly, a MEC-free MDP has MECs, just like an acyclic MDP has cycles.
```

```{prf:theorem} 
:label: thm:mdp:mecfreemaxreach
If the MDP in @thm:bellmaneq:maxreachprob is MEC-free, then the equations have a unique solution.
```
The statement indeed follows directly from @thm:finallyendcomponent. 


% Add example of an MDP with cycles but without MECs.



(sec:eliminatemecs)=
### Maximal End-Component elimination
MDPs can be transformed into a MEC-free MDP while preserving _either_ min or max reachability probabilities for a fixed target. 

```{attention}
Content missing.
```

# Algorithms for reachability probabilities

## Policy enumeration
As memoryless deterministic policies suffice, we can compute minimal and maximal reachability probabilities by simply enumerating over all memoryless deterministic policies.
We note, however, that this is exponential in the number of states and therefore highly impractical.

## Policy iteration
```{attention}
All content here is still missing. 
```

## Linear programming
The next algorithm simply reduces the problem of computing reachability probabilities into a linear programming problem. 
### Short recap
Linear programming refers to an optimization problem (a mathematical program) of a particular form.
```{prf:definition} Linear programming
Given $n$ real-values variables $x_1, \dots, x_n$  and constants $a_{ij}$, $b_i$, $c_j$, with $1 \leq i \leq m$ and $1 \leq j \leq n$, 
a linear program is of the form:
\begin{align*}  \text{Maximize}\quad & c_1 \cdot x_1 + \dots c_n \cdot x_n \\
 & \text{such that } \\
		& a_{11} \cdot x_1 + a_{12}  \cdot x_2 + \dots a_{1n}  \cdot x_n \leq b_1\\
		& \dots \\
		& a_{m1} \cdot x_1 + a_{m2}  \cdot x_2 + \dots a_{mn} \cdot x_n \leq b_m
\end{align*}
```
We can support minimization and $\geq$ by adequately negating constants.
Linear programming is supported by a rich theory. Here, it is important to note that linear programming admits for polynomial time solutions and there is very mature (academic and commercial) tool support for solving linear programs.

### Reduction
We again consider first the minimal reachability probability and then the maximal reachability probability.
#### Minimal reachability probabilities
The essence of the [Bellman equations](#thm:bellmaneq:minreachprob) is that in every state, we can pick a minimizing action.
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
The linear program has a unique solution which matches the solution we want. 

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
Strictly, $2^S$ denotes the functions from $S$ to $\{ 0, 1 \}$, i.e., functions that determines membership for every state.
Likewise, $[0,1]^S$ denotes the functions from $S$ to $[0,1]$, i.e., functions that assign a probability to every state. Assuming a total order on states, $[0,1]^S$ can be interpreted as $|S|$-dimensional vectors, which is also convenient in examples.
```
To formally describe the iterative update of the reachability probabilities, we use _Bellman operators_. 
Where _Bellman equations_ describe the optimal solution, the Bellman operators are formal [operators](#app:fixpointoperators) describe an update of values. 
We first present the operator for the minimal reachability probabilities, and then for maximal reachability probabilities.

### Minimal reachability probabilities
Recall the [Bellman equations](#thm:bellmaneq:minreachprob).
```{prf:definition} Bellman operator (MinReachProb)
:label: def:bellmanop:minreachprob
The Bellman operator for minimal reachability probabilities and for a fixed MDP is a mapping $\Phi \colon [0,1]^S \rightarrow [0,1]^S$ s.t. 
$$\Phi(F)(s) = \begin{cases} 
 	1 & \text{if }s \in T, \\
 	0 & \text{if }s \in \Szero, \\ 
\min_{a \in A(s)} \sum_{s' \in S} P(s,a,s') F(s') & \text{otherwise.}
 \end{cases}
$$
```
````{prf:example}
:label: ex:bellmanop:minreachparker

Recall the Bellman equations from @ex:bellman:minreachparker. We now show the Bellman operator:
```{code-cell} python
:tags: [remove-input]
equations = bellman.minreachprob(mdp_parker, "s2", operator=True)
Math(r"\\".join([sympy.latex(eq) for eq in equations]))
```
Let us execute the Bellman operator, maybe first on the bottom element of the lattice, which assigns every state to zero.
```{code-cell} python
:tags: [remove-input]
operator = bellman.make_operator_minreachprob(mdp_parker, "s2")
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
The essence of value iteration is thus to start with $F \gets \mathbf{0}$ and iterate until termination condition some $F \gets \Phi(F)$.
The problem is that we may not reach the fixpoint (the $n$-step reachability may always be smaller than the unbounded reachability).
What we realistically hope achieve is $$|F - V^{*}| \leq \mathbf{\epsilon} \quad \text{(pointwise)}$$.

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

```{attention}
Content is still missing.  
```

#### Interval iteration algorithm

```{attention}
Content is still missing.  
```



### Maximal reachability probabilities
```{prf:definition} Bellman operator (MaxReachProb)
:label: def:bellmanop:maxreachprob
The Bellman operator for maximal reachability probabilities (MaxReachProb) and for a fixed MDP is a mapping $\Phi \colon [0,1]^S \rightarrow [0,1]^S$ s.t. 
$$\Phi(F)(s) = \begin{cases} 
 	1 & \text{if }s \in T, \\
 	0 & \text{if }s \in \Smaxzero, \\ 
\max_{a \in A(s)} \sum_{s' \in S} P(s,a,s') F(s') & \text{otherwise.}
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

The application of value iteration is therefore similar as before. Iterating from $\mathbf{0}$ yields a correct result. 
In contrast to the minimal reachability probability, it is important how we initialize the value iteration as there are multiple fixpoints.
The simple interval iteration therefore also does not converge and one must ensure that we converge against the least fixed point. 
The standard way to ensure this is by [eliminating maximal end components](#sec:eliminatemecs). 

```{attention}
Examples are still missing.  
```
## Dynamic programming
We briefly discuss a dynamic programming approach for [acyclic MDPs](#def:mdp:acyclic). Note that by definition, acyclic MDPs are also MEC-free. 
We therefore only discuss the minimal reachability probability case here, the maximal reachability probabilities can be computed exactly analogously. 
First, let us note that on acyclic models, VI terminates with the exact solution after $h$ steps, where $h$ is the length of the longest path to a sink state (without looping in the sink states).
This yields a quadratic run time for value iteration. By using dynamic programming, we can reduce the quadratic run time to a linear run time.

In particular, if we sort the MDP topologically (i.e., by ascending distance to the target), 
the following dynamic programming scheme yields the correct solution.

````{prf:algorithm}
**Inputs** A bellman operator and topologically sorted states $\mathsf{sorted}(S)$.

**Outputs** The least fixed point of that operator.

for every $s \in \mathsf{sorted}(S)$:<br/>
<span style="margin-left: 2em;">$V(s) \gets \Phi(V)(s)$

````




```{prf:theorem}
Computing (minimal or maximal) reachability probabilities in acyclic MDPs can be done in linear time (in the size of the MDP).
```


## Combining methods
Beyond the three main families of algorithms, one can combine different algorithms.
The most relevant one is VI2PI: First VI, then PI. This yields the speed of VI (?!) and ensure an exact result with PI. 



# From reachability to temporal properties (and back)
So far, our exposition focussed exclusively on reachability probabilities. 
In particular, we categorized paths as good or bad, solely depending on whether these paths visit a particular state.
In this section, we support more general properties on paths.

## DFA properties
In a first step, we define paths that are accepted by a DFA.
More precisely, to support such properties, we label states with propositions.
```{prf:definition}
An MDP with states $S$ can be labelled by defining:
- a finite set of atomic propositions $\AP$,
- a labelling function $L\colon S \rightarrow 2^\AP$.
```
````{prf:example} 
We consider the following small MDP.
```{code-cell} python
:tags: [remove-input]
from stormvogel.examples.minitown import create_minitown_mdp

mdp = create_minitown_mdp()
sv.show(mdp)
```
````

````{prf:example} 
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

To formalize what we want, we first lift paths to traces over these executions:

(def:mdp:aptrace)=
The $\AP$-trace of a path $\xi = s_0a_0s_1a_1 \dots$ omits the actions and lifts states to the labels: $$ \aptrace(\xi) = L(s_0)L(s_1) \dots \in \big({2^\AP}\big)^{*}$$.
The set of all $\AP$-traces is called $\ApTrace$.

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
Note however, that memoryless policies are not sufficient:
```{prf:lemma}
There exist MDPs and DFAs such that:
1. $$\sup_{\pi \in \Policies} \pr^\pi(\lozenge T) > \max_{\pi \in \MdPolicies} \pr^\pi(\lozenge T) $$
2. $$\inf_{\pi \in \Policies} \pr^\pi(\lozenge T) < \min_{\pi \in \MdPolicies} \pr^\pi(\lozenge T) $$
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
Specifically, the state space contains the MDP state and the DFA state. In every transition, we update the DFA state by reading the label of the new MDP state.

In the product, we can then define target states as states that correspond to an accepting state in the DFA.
```{prf:theorem}
For any MDP $\mdp$ with states $S$, DFA property $\dfa$ with accepting states $\Acc$, and $\opt \in \{ \min, \max \}$
$$ \pr^{\opt}_\mdp(s \models \dfa) = \pr^{\opt}_{\mdp \otimes \dfa}(\lozenge \{ \langle s,q \rangle \mid q \in \Acc  \}). $$
```

````{prf:example} 
```{code-cell} python
:tags: [remove-input]

sv.show(dfa.product(mdp, aut))
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
In a _step-bounded_ property with horizon $h$ asks for reaching a target state within $h$ steps.
A DFA for this property contains of $h+2$ states: 
We start in a state encoding that have made zero steps so far. 
If a target state is visited in the MDP, we go to an absorbing and accepting state. 
Otherwise, we go from the state encoding $i$ steps so far to $i+1$ steps so far.
The state encoding $h+1$ steps is absorbing and should be interpreted as having exceeded the horizon.
Step-bounded properties can be generalized to [cost-bounded properties](#sec:costbounded).


## Büchi properties
Reachability and DFA properties consider paths up to a some (indefinite) horizon. 
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
$$ \pr^\pi( \xi \in \Paths \mid \!\!\!\underbrace{\mathsf{inf}(\xi)}_{\text{choices visited inf often}}\!\!\! \text{ is an EC } ) = 1	$$
```
The proof first observes that for any infinite path in a finite MDP, there must be at least one choice which is visited infinitely often.
Every successor state of that choice must also be visited infinitely often, and thus in those states, also a choice is visited infinitely often. 
This ascending set then stabilises with a strongly connected component where a policy can stay forever, i.e., an EC.

```{prf:definition}
An EC satisfies a B\"uchi property $\varphi$ if it contains a target state. 
```
We justify this wording by remarking that a policy can choose to visit any state in an EC infinitely often. In particular, such a policy can thus satisfy $\varphi$.

```{prf:theorem} 
- Let $U_\varphi$ be the union of all $S'$ in an EC that satisfies the Büchi property $\varphi$. 	
- Let $V_\varphi$ be the union of all $S'$ in an EC that does not satisfies the Büchi property $\varphi$.
Then:
1. $\pr^{\max}(\varphi) = \pr^{\max}(\lozenge U_\varphi)$
2. $\pr^{\min}(\varphi) = 1 - \pr^{\max}(\lozenge V_\varphi)$.	
```
The theorem above gives rise to a (naive) algorithm: Compute all ECs, classify each EC as satisfying or not satisfying, and then solve a reachability probability problem. 
Note that such an algorithm is not efficient, as there are exponentially many ECs. However, careful graph algorithms avoid the necessity to precompute all ECs, while in spirit coinciding with the naive algorithm.


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

(def:rewardpath)=
Given a path $\xi = s_0a_0s_1\dots$, the reward of $\xi$ is defined as $$\mathsf{rew}(\xi) = \sum_i = r(s_i, a_i)$$. 


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
For MDPs, we can study reachability rewards under a minimizing or a maximizing policy. We write $\mathbb{E}^{\pi}(\lozenge T)$ for the expected reward in the Markov chain induced by $\pi$.
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
\max_{a\in A(s)} \operatorname{rew}(s,a) + \sum_{s'}P(s,a,s') \cdot F(s')
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

### Optimal discounted rewards in MDPs
We can now define $\mathbb{E}^{\max}(s \models \mathsf{tot}_\gamma)$ and $\mathbb{E}^{\min}(s \models \mathsf{tot}_\gamma)$
analogously to $\pr^{\min}$ and $\pr^{\max}$. As before, memoryless deterministic policies suffice.

#### Bellman operator
Likewise, we can define Bellman equations and the operators. For conciseness, we only define the operator:
```{prf:definition} Bellman operator (MaxDiscReward/MinDiscReward)
:label: def:bellmanop:mdiscountedexpreward
The Bellman operator for maximal discounted total reward and for a fixed MDP is a mapping $\Phi_{\drew_\gamma}^{\max} \colon [0,\infty)^S \rightarrow [0,\infty)^S$ s.t. 
$$\Phi_{\drew_\gamma}^{\max}(F)(s) = 
\max_{a \in A(s)} r(s,a) + \gamma \cdot \sum_{s' \in S} P(s,a,s') F(s').
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



(sec:costbounded)=
## Cost-bounded reachability 
```{attention}
All content here is still missing.
```