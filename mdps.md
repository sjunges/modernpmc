---
numbering:
  heading_1: true
  heading_2: true
  heading_3: true

kernelspec:
  name: python3
  display_name: Python 3
---


# Markov Decision Processes
```{attention}
Examples are still missing.
```

```{code-cell} python
import stormpy
import stormpy.info
stormpy.info.storm_version()
```


```{code-cell} python
import stormvogel
vis = stormvogel.show(
    stormvogel.examples.create_car_mdp()
)
```

# What are Markov decision processes?

```{prf:definition} Markov Decision Process
An _MDP_ $\mdp$ is a tuple

$$
\mdptuple
$$

where

- $S$ is a nonempty set of _states_,
- $A$ is a nonempty set of _actions_,
- $\delta \colon S \times A \nrightarrow \Distr{S}$ is a _partial transition function_.
```

Additionally, we often assume the existence of a unique _initial state_ $\sinit$.
Later we extend MDPs with _atomic propositions_, _target states_, and _reward functions_.

```{prf:definition} Enabled actions
For any state $s$, the set of _enabled actions_ is

$$
\EnAct{s} = \{ a \mid \delta(s,a) \neq \bot \}.
$$
```

We use the notion of a _choice_ to denote a state–action pair
$\langle s,a \rangle$ where $a \in \EnAct{s}$.

Markov chains are MDPs with exactly one choice per state. We can then omit actions and write 
a Markov chain as a tuple $\langle S, \delta \rangle$, potentially extended by initial states.

For any choice $\langle s,a\rangle$ we may write $\delta(s,a,s')$ to denote $\delta(s,a)(s')$.

```{warning} Assumptions
Unless stated otherwise we assume

- $S$ and $A$ are finite,
- there are no deadlocks: $\EnAct{s} \neq \emptyset$, and
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
we write $\last(\xi)$ for the final state.

### Policies

```{prf:definition} Policy
For an MDP $\mdp$, a _policy_ is a function

$$
\pi \colon \Paths^\mdp \rightarrow \Distr{A}.
$$

- A policy $\pi$ is _memoryless_ if
$
\pi(\xi) = \pi(\xi')
$
for all paths with the same last state.
- A policy is _deterministic_ if every distribution $\pi(\xi)$ is Dirac.
```

We denote the set of all policies with $\Policies$ and use $\MdPolicies$ for the memoryless deterministic policies

Memoryless deterministic policies can be written as
$
\pi : S \rightarrow A.
$

### Induced Markov Chains

Once a policy resolves nondeterminism in an MDP, the behaviour becomes purely probabilistic and can thus be captured by a Markov chain.

```{prf:definition} Induced Markov chain (general policies)
Given MDP $\mdp$ and policy $\pi$, the _induced Markov chain_
$\mdp[\pi]$
is
$
\langle \Paths^\mdp , \delta^\pi \rangle
$
with

$$
\delta^\pi(\xi)(\xi \cdot a s)
=
\pi(\xi)(a)\,\delta(\last{\xi},a)(s).
$$
```
```{prf:remark} Infinite Markov chain
The above Markov chain is infinite, but reachability probabilities remain well defined.
```
It is useful to consider a special case for memoryless (possibly randomised) policies
```{prf:definition} Induced Markov chain (memoryless policies)
For a memoryless policy $\pi$, the induced Markov chain is $\mdp[\pi] =
\langle S , \delta^\pi \rangle$
where
$$ \delta^\pi(s)(s') = \sum_a \pi(s)(a)\delta(s,a)(s'). $$
```

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

### Possible reachability
One of the simplest questions we can ask is to find the set of states that can reach the targets with positive probability.

More formally, we denote this set of states  $$\Spos = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) > 0 \}$$, i.e., the set of states where the _maximum_ probability over all policies is positive. 
It holds that $\Spos = S \setminus \Smaxzero$ where $\Smaxzero = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 0 \}$. 
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

### Almost-sure reachability
We are also interested in computing the set of states from which it is possible to ensure that we almost-surely reach the target states.

More formally, we denote this set of state $$\Smaxas = \{ s \mid \exists \pi \text{ s.t. } \pr^\pi(s \models \lozenge T) = 0 \}.$$

We use a recursive equation to characterize this set. If $s \in T$, then clearly $s \in \Smaxas$. Otherwise, for $s \not \in T$:  $$s \in \Smaxas \text{\quad iff \quad} \exists a \in \EnAct{s}. \forall s' \in \supp{\delta(s,a)}. s' \in \Smaxas. $$

To compute a the set $\Smaxas$, we want to provide an iterative procedure, which operates on sets of states.
Specifically, we define the operator

$$
\mathsf{AS}\colon 2^S \rightarrow 2^S
$$
such that
$$
\mathsf{AS}(X)=
\{ s \mid
\exists a\in\EnAct{s}.
\forall s'\in\supp{\delta(s,a)}.
s'\in X \}\cup T
$$
The operator is monotonic, the lattice is finite. The greatest fixpoint of this operator yields the solution. 

```{attention}
The explicit algorithm is still missing.
```

## Quantitative reachability

We compute

* maximal reachability probability

$$
\sup_{\pi} \pr^\pi(\lozenge T)
$$

* minimal reachability probability

$$
\inf_{\pi} \pr^\pi(\lozenge T)
$$

### Minimal reachability probability

```{prf:theorem} Bellman equations (MinReachProb)
:label: thm:bellmaneq:minreachprob
Consider variables $x_s$ for each $s \in S$.
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
```{hint} Bellman equations vs Bellman operators
We will later also talk about Bellman operators. 
At that point, we will clarify the difference.
```
```{hint} Different Bellman equations 
We will see various different Bellman equations. It is important to clarify which Bellman equations one is talking about.
```

We observe the following:
- If there is only one choice per state, the equation system trivally reduces to the linear equation system for Markov chains.
- Using the preprocessing with $\Smaxas$, we could extend the set of states for which we can ensure $x_s = 1$.
- We must ensure that states that cannot reach the target have a zero probability.

### Maximal reachability probability
While most aspects of computing minimal and maximal reachability probabilities are analogously, the theorem about the Bellman equations differ for both cases.
```{prf:theorem} Bellman equations (MaxProb)
Consider variables $x_s$ for each $s \in S$.
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
```{attention}
Relate to MECs
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

## Maximal end components
```{attention}
All content here is still missing. 
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
The essence of the Bellman equations is that in every state, we can pick a minimizing action.
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
The adaptions are straightforward. What is notable is that the Bellman equations require that we find the smallest solution, which we do by minimising here.
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
More formally, we move from the lattice $(2^S, \subseteq)$ where $2^S$ denotes the powerset on $S$ to the lattice $([0,1]^S, \preceq)$ where $\preceq$ denotes pointwise inequality.
```{note} Notation
Strictly, $2^S$ denotes the functions from $S$ to $\{ 0, 1 \}$, i.e., functions that determines membership for every state.
Likewise, $[0,1]^S$ denotes the functions from $S$ to $[0,1]$, i.e., functions that assign a probability to every state. Assuming a total order on states, $[0,1]^S$ can be interpreted as $|S|$-dimensional vectors, which is also convenient in examples.
```
To formally describe the iterative update of the reachability probabilities, we use _Bellman operators_. 
Where _Bellman equations_ describe the optimal solution, the operators describe an update of values. 
We first present the operator for the minimal reachability probabilities, and then for maximal reachability probabilities.

### Minimal reachability probabilities
Recall @thm:bellmaneq:minreachprob.
```{prf:definition} Bellman operator (MinReachProb)
The Bellman operator for minimal reachability probabilities and for a fixed MDP is a mapping $\Phi \colon [0,1]^S \rightarrow [0,1]^S$ s.t. 
$$\Phi(F)(s) = \begin{cases} 
 	1 & \text{if }s \in T, \\
 	0 & \text{if }s \in \Szero, \\ 
\min_{a \in A(s)} \sum_{s' \in S} P(s,a,s') F(s') & \text{otherwise.}
 \end{cases}
$$
```
After looking at a concrete example, let us consider a more generic setting.
$\Phi(\mathbf{0})$ yields the indicator function for $T$, $\mathbb{1}_T$.
$\Phi(\mathbb{1}_T)$ yields the one-step minimal reachability probabilities to reach $T$. 
By simple substitution, we have $\Phi(\Phi(\mathbf{0})) = \Phi(\mathbb{1}_T)$.
Likewise, $\Phi(\Phi(\Phi(\mathbf{0})))$, denoted $\Phi^3(\mathbf{0})$ yields the two-step minimal reachability probabilities and indeed,
$\Phi^{n+1}(\mathbf{0})$ denotes the $n$-step minimal reachability probabilities.

```{prf:lemma}
The Bellman operator for MinReachProb $\Phi$ is monotonic and $\omega$-continuous.
```


### Maximal reachability probabilities
```{prf:definition} Bellman operator (MaxReachProb)

```{attention}
Any content here beyond the definition of the Bellman operators is still missing.  
```

# From reachability to temporal properties (and back)
```{attention}
All content here is still missing. 
```
## DFA tasks

## Buechi tasks

## Towards LTL and $\omega$-regular

# Rewards
```{attention}
All content here is still missing.
```
## Minimal expected rewards
## Maximal expected rewards
## Discounted rewards (max and min)