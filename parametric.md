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

# Parametric Markov Decision Processes
In this chapter, we discuss parametric MDPs. 


```{code-cell} python
import stormpy
import sympy
from fractions import Fraction
import stormvogel as sv
import stormvogel.to_dot
import stormvogel.parametric.region
import stormvogel.teaching as teach
import stormvogel.bird as bird
sympy.init_printing()
from IPython.display import Math
import stormvogel.teaching.bellman as bellman
import stormvogel.examples as examples
```


# What are parametric Markov decision processes?
Parametric MDPs describe MDPs where the probabilities are not a fixed number, 
but rather a symbolic expression over some parameters.
```{prf:definition} Parametric MDPs
A parametric Markov Decision Process (pMDP) is a tuple 
$$ \langle S, A, \mathbf{x}, \delta \rangle $$
 where $S$ is a finite and nonempty set of states, $A$ is a finite and nonempty set of actions, 
 $\mathbf{x}$ is a finite and nonempty set of _parameters_, and the transition relation $\delta$ is given by $\delta \colon S  \times A \nrightarrow S \rightarrow \ratfunc{\mathbf{x}}$.
```
Choices are thus not labeled by distributions over states, but by assigning to (successor) states rational functions over some set of parameters.
pMDPs can be extended with initial states, labels, target states, etc., just like standard (parameter-free) MDPs. 

Parametric Markov chains are pMDPs such that in every state, only one action is enabled. 
We typically denote these as tuples $\langle S, \mathbf{x}, \delta \rangle$.

````{prf:example} Von Neumann Unbiased Coins 
:label:ex:pmc:vonNeumannTrick

Say we want to simulate an unbiased, perfectly random coin flip.
We only have access to an old, biased coin and can flip it as many times as we want —
that is, we have access to an infinite stream of biased random bits,
each of which is $0$ with some unknown but fixed probability $0<p<1$.
A simple solution @vonneumannVariousTechniques1951 : 
Extract the first two bits from the stream; if they are different, return the value of the first; otherwise try again.
The following pMC models the protocol. We flip a coin, and obtain a $0$ with probability $p$ and a $1$ with probability $1-p$.
```{code-cell} python
:tags: [remove-input]
# Create MDP
vnpmc = sv.model.new_dtmc()
p = vnpmc.declare_parameter("p")
sinit = vnpmc.initial_state
sinit.set_friendly_name("_")
# Add states with friendly names
s0 = vnpmc.new_state(friendly_name="0")
s1 = vnpmc.new_state(friendly_name="1")
s10 = vnpmc.new_state("ret0", friendly_name="01")
s01 = vnpmc.new_state("ret1", friendly_name="10")
sinit.set_choices([(p,s0), (1-p,s1)])
s0.set_choices([(p,sinit), (1-p,s01)])
s1.set_choices([(p,s10), (1-p,sinit)])

# Add sink self-loops (important for well-formed MDP)
vnpmc.add_self_loops()
stormvogel.to_dot.plot_model_pydot(vnpmc, positions={sinit: (0,0), s0: (2,0), s01: (4,0), s1: (-2,0), s10: (-4,0)}, state_colors={"A": "red", "B": "blue"}, default_fill="white")
```
In this chapter, we demonstrate that the probability to reach state $01$ is $0.5$, irrespectively of the value of $p$.
We note that such an analysis is not possible with the standard s-a-rectangularity assumption in robust models.
````

````{prf:example} Parametric Knuth-Yao Dice
:label:ex:parametric:kydie
The following example shows the Knuth-Yao die[^kydievariants], where alternatingly two coins are flipped. 
The first has a bias $x$, the second a bias $y$. 
```{code-cell} python
:tags: [remove-input]
pkydie = examples.create_knuth_yao_pmc_twocoins()
stormvogel.to_dot.plot_model_pydot(pkydie)
```
Analysing the effect of $x$ and $y$ on the probability to reach, say, `rolled3` is non-trivial.
Consider, e.g., the question of how to get this probability above a threshold, i.e., how to maximise the reachability probability.
Clearly, $x$ should not be so small that the probability mass at the initial state flows into the right branch. 
On the other hand, $x$ should also not be too large, as then, `rolled3` cannot be reached.
The precise optimum depends on the influence of $s_3$, which itself depends on the value of $y$. 
````
[^kydievariants]: We note that the precise transition relations differ over the literature: In particular, $x$ and $1-x$ is sometimes swapped in places.

```{admonition} Note
While we discuss the concepts generally for pMDPs, most examples and the algorithms later in this chapter focus on pMCs.
```

The intention of pMDPs is that by substituting parameters with concrete values, one obtains an MDP. 
Substitutions of parameters by such values are called well-defined valuations.
```{prf:definition} Well-defined valuations
Given a pMDP $\pmdp = \langle S, A, \mathbf{x}, \delta \rangle$ over parameters $\mathbf{x}$. 
A valuation $\val$ is a mapping $\mathbf{x} \rightarrow \mathbb{R}$.
A valuation $\val$ is _well-defined_, if 
1. $\delta(s,a)(s')[\val] \geq 0$ for all $s, s' \in S$ and $a \in \EnAct{s}$. 
2. $\sum_{s'\in S} \delta(s,a)(s')[\val] = 1$ for all $s \in S$, $a \in \EnAct{s}$. 
```
````{prf:example}
For the [Knuth-Yao die](#ex:parametric:kydie), the well-defined valuations are all valuations with $x,y \in [0,1]$. 
For the following example, the well-defined valuations must assign $y \in [0,2]$ and $x+z=1$.
```{code-cell} python
:tags: [remove-input]
from stormvogel.parametric.region import plot_annotated_regions_1d                                                                                                                                                                                                                                                                    
pmc = sv.model.new_dtmc(create_initial_state=False)                                                                                                                        
x = pmc.declare_parameter("x")                                                                                                                                       
y = pmc.declare_parameter("y")
z = pmc.declare_parameter("z")
                                                                                                                                                  
t0 = pmc.new_state(["init"], friendly_name="s0")                                                                                                                     
t1 = pmc.new_state(friendly_name="s1")                                                                                                                             
t2 = pmc.new_state(["T"], friendly_name="s2")   
                                                                                                                                                                     
pmc.set_choices(t0, [(0.5*y,     t1), (1 - 0.5*y, t2)])                                                                                                                      
pmc.set_choices(t1, [(x+z, t2)])                                                                                                                  
pmc.set_choices(t2, [(1,     t2)])   
stormvogel.to_dot.plot_model_pydot(pmc)
```
````
Any pMDP together with a well-defined valuation thus describes an MDP.
```{prf:definition} Induced MDP
Given a pMDP $\pmdp = \langle S, A, \mathbf{x}, \delta \rangle$ and a well-defined valuation $\val$,
the induced MDP $\pmdp[\val]$ is the MDP $\langle S, A, \delta' \rangle$,
where $\delta'(s,a)(s') = \delta(s,a)(s')[\val]$ for all $s,s'$ and $a \in \EnAct{s}$.
```
````{prf:example} Induced model
The following instantiates the pMC from @ex:pmc:vonNeumannTrick with $p=\frac{1}{3}$.
```{code-cell} python
:tags: [remove-input]
induced = vnpmc.get_instantiated_model({"p": Fraction(1,3)})
stormvogel.to_dot.plot_model_pydot(induced, positions={induced.get_state_by_id(sinit.state_id): (0,0), 
     induced.get_state_by_id(s0.state_id): (2,0), induced.get_state_by_id(s01.state_id): (4,0), induced.get_state_by_id(s1.state_id): (-2,0), induced.get_state_by_id(s10.state_id): (-4,0)})
```
````
An important distinction can be made based on the graph-structure of the pMDP $\pmdp$ and the induced MDP $\pmdp[\val]$.
Valuations where these coincide are called _graph-preserving_.
```{prf:definition} Graph-preserving valuations
Given a pMDP $\pmdp = \langle S, A, \mathbf{x}, \delta \rangle$ over parameters $\mathbf{x}$. 
A valuation $\mathbf{x} \rightarrow \mathbb{R}$ is _graph preserving_, if 
$$ \delta(s,a)(s')[\val] > 0 \text{ iff }\delta(s,a)(s') \neq 0\quad\text{for all }s, s' \in S\text{ and }a \in \EnAct{s}.$$ 
```

We are often interested in sets of valuations, 
which are commonly referred to as regions (based on a geometric interpretation).
```{prf:definition} Well-defined regions
A set of valuations for a pMDP is called a region. 
The _well-defined region_ is the set of all well-defined valuations.
A region is _well-defined_, if it is a subset of the well-defined region.
```
```{warning} Assumptions
We only consider pMDPs with non-empty well-defined regions. 
Deciding whether this set is empty is hard in general, but it is easy, e.g., if all transitions are affine expressions.
```
Regions can come in different shapes. 
Here, we are particularly interested in rectangular regions.
```{prf:definition} Rectangular regions
A region $R$ is _rectangular_, if there exist $l_x, u_x$ for every $x \in \mathbf{x}$, 
such that $$R = \{ \val \mid \text{for all } x \in \mathbf{x}: l_x \leq \val(x) \leq u_x \}.$$ 
We write $R(x)$ for the interval $[l_x, u_x]$.
```
```{prf:remark} Rectangularity
The notion of rectangularity is not (directly) related to notions like (s,a)-rectangularity for robust MDPs/interval MDPs.
```
Any pMDP together with a region of well-defined valuations describes a set of MDPs.

```{prf:definition} Generated Markov chains.
Given a pMDP $\pmdp$ and a well-defined region $R$, 
$$ \generator{\pmdp}{R} = \{ \pmdp[\val] \mid \val \in R \} $$
If $R$ is the well-defined region, we may simply write 
$ \generatorwd{\pmdp} $.
```

## Types of pMDPs
pMDPs can be distinguished on how parameters occur on transitions. 
Intuitively, _polynomial pMDPs_ have transition probabilities where the denominator is a constant,
_affine pMDPs_ are polynomial pMDPs where the (total) degree of every transition probability is at most one.
Here, we often restrict ourselves to simple pMDPs.
```{prf:definition} Simple pMDPs
A pMDP is simple, if for every choice $(s,a)$ and all $s' \in S$: $$\delta(s,a)(s') \in \mathbb{Q} \cup \{ x, 1-x \mid x \in \mathbf{x} \}$$
and additionally
$$ \sum_{s' \in S} \delta(s,a)(s') = 1. $$
```
As a consequence, non-constant transition probabilities always occur in pairs $x$, $1-x$ for $x \in \mathbf{x}$.
The models in @ex:pmc:vonNeumannTrick and @ex:parametric:kydie are indeed simple pMCs.
As we will see below, except for the syntactical form, algorithmically, there is nothing simple about simple pMDPs.
A key characteristic of simple pMDPs, and the main reason we use them, is that the set of well-defined regions is simply the unit hypercube and in particular rectangular.

## Relation to interval MDPs
Parametric MDPs are closely related to non-rectangular robust MDPs.
The syntax of parametric MDPs already naturally indicates a static type of uncertainty.
As already highlighted in @ex:pmc:vonNeumannTrick: in contrast to (s,a)-rectangular robust MDPs, parametric MDPs can express that two different transitions must be the same.

The following lemmas between parametric MDPs with rectangular regions and interval MDPs formalizes the relation and can be adapted for various other robust models.
Basically, we obtain an iMDP by replacing, for every parameter $x$, 
every occurrence of $x$ with $[l_x, u_x]$ and every occurrence of $1-x$ by $[1-u_x, 1-l_x]$.
```{prf:definition} IMDP abstraction
Given a simple pMDP $\pmdp = \langle S, A, \delta \rangle$  with a rectangular region $R$. 
The interval MDP lifting $\pmdp$ and $R$ is given by 
$$ \abst{\pmdp}{R} = \langle S, A, \delta' \rangle $$
with $$ \delta'(s,a)(s') = \delta(s,a)(s')[x_1 \gets R(x_1), \dots , x_n \gets R(x_n)]$$ 
using an interval extension of substitution.
```

````{prf:example}
We provide an example for @ex:parametric:kydie and $R$ with $$ \frac{1}{2} \leq R(x) \leq \frac{3}{5}, \frac{3}{7} \leq R(y) \leq \frac{4}{7}.$$
```{code-cell} python
:tags: [remove-input]
region = sv.parametric.region.RectangularRegion({"x": tuple([Fraction(1,2), Fraction(3,5)]), "y": tuple([Fraction(3,7), Fraction(4,7)])})
imc = sv.parametric.region.to_interval_mdp(pkydie,region)
stormvogel.to_dot.plot_model_pydot(imc)
```
````

The IMDP is a proper abstraction, when considered via the set of MDPs that can be generated. 
```{prf:lemma} iMDPs abstract pMDPs with rectangular regions
:label:lem:par:lifting
Given a simplex pMDP $\pmdp$ with a rectangular region $R$. 
Then $\generator{\pmdp}{R} \subseteq \generatorint{\abst{\pmdp}{R}}$. 
```
````{prf:example} iMDPs abstract pMDPs
Consider $\pmdp$ as in @ex:pmc:vonNeumannTrick and $R$ with $0.4 \leq R(p) \leq 0.7$. 
Let $\imdp'$ be the lifting of that pMC and $R$.
The following MC is in $\generatorint{\imdp'}$ but not in $\generator{\pmdp}{R}$.
```{code-cell} python
:tags: [remove-input]
# Create MDP
vnpmccp = sv.model.new_dtmc()
p = vnpmccp.declare_parameter("p")
sinit = vnpmccp.initial_state
sinit.set_friendly_name("_")
# Add states with friendly names
s0 = vnpmccp.new_state(friendly_name="0")
s1 = vnpmccp.new_state(friendly_name="1")
s10 = vnpmccp.new_state("ret0", friendly_name="01")
s01 = vnpmccp.new_state("ret1", friendly_name="10")
sinit.set_choices([(0.5,s0), (0.5,s1)])
s0.set_choices([(0.4,sinit), (0.6,s01)])
s1.set_choices([(0.7,s10), (0.3,sinit)])

# Add sink self-loops (important for well-formed MDP)
vnpmccp.add_self_loops()
stormvogel.to_dot.plot_model_pydot(vnpmccp, positions={sinit: (0,0), s0: (2,0), s01: (4,0), s1: (-2,0), s10: (-4,0)}, state_colors={"A": "red", "B": "blue"}, default_fill="white")
```
````
Note that vice versa, we can express every interval MDP as a parametric MDP. 
```{prf:definition} pMDP representation for iMDPs
Given an interval MDP $\imdp = \langle S, A, \delta \rangle$, 
we define the underlying variables  $$\mathbf{x} = \{ x_{(s,a,s')} \mid s,s' \in S, a \in A \}.$$
For $\imdp$, we then define 
- the _matching pMDP_ $\mathsf{parametric}(\imdp) = \langle S, A, \mathbf{x}, \delta' \rangle$,  with 
$$ \delta(s,a)(s') = \begin{cases} x_{s,a,s'} & \text{if } \delta(s,a)(s') > 0  \\ 0 & \text{ otherwise.} \end{cases}$$
- and the _matching region_ $R$ with $$ R(x_{(s,a,s')}) = \delta(s,a)(s'). $$
```
```{prf:lemma} Matching pMDPs generate the same MDPs as iMDPs
Let $\imdp$ be an iMDP and $\pmdp'$ the matching pMDP and $R$ the matching region. Then:
$$ \generatorint{\imdp} = \generator{\pmdp'}{R}. $$
```

(sec:pmc:representingpolicies)=
## Relation to policy-induced models
Given an MDP $\mdp$[^representingpolicies:extendedtoparametric], we can represent an [infinite set of induced Markov chains](#def:mdps:generatorinduced) of a memoryless, randomising policy via $\generatormr{\mdp}$. 
This Markov chain is obtained by picking weights for every state-action combination, $w_{s,a}$.
```{prf:definition} Corresponding pMC for MDP
Given an MDP $\mdp = \langle S, A, \delta \rangle$, 
we define the underlying variables  $$\mathbf{x} = \{ w_{(s,a)} \mid s \in S, a \in \EnAct{s} \}.$$
For $\mdp$, we then define the _corresponding pMC_ $\mathsf{corresponding}(\mdp) = \langle S, A, \mathbf{x}, \delta' \rangle$,  with 
$$ \delta(s,a)(s') = \begin{cases} \sum_{a} w_{s,a} \cdot \delta(s,a)(s') & \text{if } a \in \EnAct{s}  \\ 0 & \text{ otherwise.} \end{cases}$$
```
From the definitions, the following follows immediately:
```{prf:lemma}
For any MDP:
$$ \generatormr{\mdp} = \generatorwd{\mathsf{corresponding}(\mdp)}. $$
```
The value of this construction is twofold:
- We can mildly modify it to add additional constraints on the policies, which then translate to constraints on the parameter region of interest.
- We can rephrase problems on pMDPs as problems on pMCs.

[^representingpolicies:extendedtoparametric]: The construction can be easily lifted to parametric MDPs. We omit that for conciseness.


# Solution functions and their computation
The value of an MDP is a useful abstraction to refer to a quantity of interest in parameter-free models.
In parametric models, the value is no longer a constant but rather a function over the parameters.

Specifically, for any fixed pMDP, every valuation maps to an MDP.
For any fixed objective (such as maximal reachability objectives), an MDP maps to a value.
Solution functions combine this by fixing the pMDP and the objective.
That is, the solution function (for a pMDP and an objective) maps every valuation to a value,
i.e., solution functions are expressions over the parameters. 
By definition, substituting the parameters in the solution function yields the same result as substituting the parameters in the pMDP and applying model checking.
In what follows, we only study solution functions for maximal reachability probabilities. 
```{prf:definition} Solution function (MaxReachProb)
Given a pMDP $\pmdp$ with initial states and targets $T$, and a well-defined region $R$.
The solution function  $\solfunc\colon R \rightarrow [0,1]$ is defined as $\solfunc(\val) = \pr^{\max}_{\pmdp[\val]}(\lozenge T)$.
```
````{prf:example}
Consider the following pMC.
```{code-cell} python      
:tags: [remove-input]                                                                                                                                                                                                                                                                         
pmc = sv.model.new_dtmc(create_initial_state=False)                                                                                                                        
x = pmc.declare_parameter("x")                                                                                                                                       
                                                                                                                                                                     
s0 = pmc.new_state(["init"], friendly_name="s0")                                                                                                                     
s1 = pmc.new_state(friendly_name="s1")  
s2 = pmc.new_state(friendly_name="s2")                                                                                                                               
s3 = pmc.new_state(["T"], friendly_name="s3")                                                                                                                               
s4 = pmc.new_state(["sink"], friendly_name="s4")
                                                                                                                                                                     
pmc.set_choices(s0, [(x,     s1), (1 - x, s4)])                                                                                                                      
pmc.set_choices(s1, [(1 - x, s2), (x,     s4)])
pmc.set_choices(s2, [(x,     s3), (1 - x, s4)])                                                                                                                      
pmc.set_choices(s3, [(1, s3)])          
pmc.set_choices(s4, [(1, s4)])  
stormvogel.to_dot.plot_model_pydot(pmc)
```
The solution function is {eval}`str(stormvogel.model_checking(pmc, "P=? [F \"T\"]").at_init())`, which is obtained by multiplying the transition probabilities along the single path from initial state to target.
````
```{prf:remark} Uniqueness of the solution functions
The literature is inconsistent wrt the definition of a solution function: There is _the_ solution function for a given region $R$, which is only defined on $R$,
or _a_ solution function that coincides with _the_ solution function on $R$ and is arbitrarily defined otherwise. 
In particular, algorithms to compute solution functions compute functions that are defined on $\mathbb{R}^m$, but which are only solution functions on the well-defined region (or a subset thereof).
```

## Shape and size of solution functions
It is sometimes interesting to reduce the domain of the solution function, e.g., to discuss the shape of the function,
e.g., to talk about the representation of solution functions on graph-preserving regions.
The shape and size of solution functions is not arbitrary. 

### Solution functions on parametric Markov chains. 
We state two key facts about solution functions for pMCs.
```{prf:theorem} Shape of solution functions
Given a pMC. The solution function for $\pmdp$ on well-defined region $R$ is guaranteed to be:
- lower semicontinuous and
- a rational function, if $R$ is graph-preserving and
- a polynomial, if $\pmdp$ is polynomial and acyclic.
```
While there are direct proofs for the latter two points, we will justify them as a consequence of the correctness of the state elimination algorithm.
The first point follows from 
1. the continuity of the rational function (notice that the denominator will never be zero on a well-defined point)
2. the fact that removing transitions can only reduce the reachability probability. 

````{prf:example}
This example demonstrates that the solution function on the well-defined region is in general not continuous. 
```{code-cell} python
:tags: [remove-input]
pmc = sv.model.new_dtmc(create_initial_state=False)                                                                                                                        
x = pmc.declare_parameter("x")                                                                                                                                       
                                                                                                                                                                     
s0 = pmc.new_state(["init"], friendly_name="s0")                                                                                                                     
s1 = pmc.new_state(friendly_name="s1")  
s0.set_choices([(1-x, s0), (x, s1)])
s1.set_choices([(1, s1)])
```
In particular, the probability to leave the initial state and thus reach the target is one, if $x \in (0,1]$. 
If $x=0$, the probability is zero.
````

```{prf:theorem} Size of solution functions
For acyclic simple pMCs with $n$ states and $k$ parameters, the size of the solution function (written as sum of terms) is polynomial in $n$ and exponential in $k$, in the worst case.
```
In particular, this result also means that every algorithm to compute solution functions is necessarily exponential in the number of parameters. 


### Solution functions on parametric Markov decision processes.
The solution function for a pMDP is then the finite maximum over these rational functions.

## Computing solution functions for pMCs
We present the state-elimination algorithm for computing solution functions. 
Note that this algorithm is not optimal in terms of complexity, in particular, it can run in exponential time for a fixed number of parameters. 
Nevertheless, it is arguably the simplest algorithm. 
The algorithm is closely connected to the state elimination algorithm for nondeterministic finite automata.

State elimination computes solution functions with a series of transformations on the parametric Markov chain.
1. **Make the target state absorbing**. This simplifies the description below.
2. **Self-loop elimination**.  
Given any state $s$ which is not a sink-state.
We can eliminate self-loops on state $s$ by rescaling all other outgoing transitions. 
3. **Transition shortcutting**.
Take a state $s'$ without self-loop.
Replace a transition from $s$ to $s'$ by new or updated transitions from $s$ to all successors of $s'$.
The probability from $s$ to $\hat{s}$, with $\hat{s}$ a successor of $s$, is updated to $$ \delta(s,\hat{s}) \gets \delta(s,\hat{s}) + \delta(s,s') \cdot \delta(s',\hat{s}).  $$
The transition probability $\delta(s,s')$ is updated to 0.
4. **State removal** (optional). 
Take any state with no incoming transition. 
Remove the state and all outgoing transitions.

None of the steps above affects the reachability probability in any Markov chain generated by the pMC. 
In particular, self-loop elimination is correct: taking a loop with transition probability $<1$ infinitely often has probability zero.

Eliminating states can be applied to any non-initial, non-sink state $s$. It runs transition shortcutting on all incoming transitions of $s$ and then optionally removes a state $s$.
The state elimination algorithm removes self-loops whenever possible and then eliminates non-initial, non-sink states.

````{prf:example}
We apply elimination steps on @ex:pmc:vonNeumannTrick, repeated here for convenience. 
The targets are already absorbing and state $s_0$ has no self-loop, so we can eliminate the transition from the initial state to $s_0$.
This introduced a self-loop at the initial state, and a transition from initial state to the state 01.
```{code-cell} python
:tags: [remove-input]
import stormvogel.teaching.parametric as elim
vnpmccopy = sv.model.new_dtmc()
pcopy = vnpmccopy.declare_parameter("p")
sinitcopy = vnpmccopy.initial_state
sinitcopy.set_friendly_name("_")
# Add states with friendly names
s0copy = vnpmccopy.new_state(friendly_name="0")
s1copy = vnpmccopy.new_state(friendly_name="1")
s10copy = vnpmccopy.new_state("ret1", friendly_name="10")
s01copy = vnpmccopy.new_state("ret0", friendly_name="01")
sinitcopy.set_choices([(pcopy,s0copy), (1-pcopy,s1copy)])
s0copy.set_choices([(pcopy,sinitcopy), (1-pcopy,s01copy)])
s1copy.set_choices([(pcopy,s10copy), (1-pcopy,sinitcopy)])
vnpmccopy.add_self_loops()
stormvogel.to_dot.plot_model_pydot(vnpmccopy,positions={sinitcopy: (0,0), s0copy: (2,0), s01copy: (4,0), s1copy: (-2,0), s10copy: (-4,0)}, default_fill="white", self_loop_position="s")
elim.eliminate_transition(vnpmccopy, sinitcopy, s0copy)
stormvogel.to_dot.plot_model_pydot(vnpmccopy,positions={sinitcopy: (0,0), s0copy: (2,1), s01copy: (4,0), s1copy: (-2,0), s10copy: (-4,0)}, default_fill="white", self_loop_position="s")
```
Now, as there are no more incoming states in $s_0$, we can eliminate the state and then also do the same elimination for $s_1$ (which also has no self-loop).
```{code-cell} python
:tags: [remove-input]
elim.eliminate_state(vnpmccopy, s0copy, remove=True)
elim.eliminate_state(vnpmccopy, s1copy, remove=True)
stormvogel.to_dot.plot_model_pydot(vnpmccopy,positions={sinitcopy: (0,0), s0copy: (2,0), s01copy: (4,0), s1copy: (-2,0), s10copy: (-4,0)}, default_fill="white", self_loop_position="s")
```
Finally, we remove the self-loop from the initial state to rescale the probabilities.
```{code-cell} python
:tags: [remove-input]
elim.eliminate_selfloop(vnpmccopy,sinitcopy)
stormvogel.to_dot.plot_model_pydot(vnpmccopy,positions={sinitcopy: (0,0), s0copy: (2,0), s01copy: (4,0), s1copy: (-2,0), s10copy: (-4,0)}, default_fill="white", self_loop_position="s")
````

````{prf:example}
We now also run state elimination for @ex:parametric:kydie.
```{code-cell} python
:tags: [remove-input]
import stormvogel.teaching.parametric as elim
pkydie_e = pkydie.copy()
states = list(pkydie_e.states)
for state in states:
 if state.is_initial() or len(list(state.labels)) > 0:
  continue
 if state.has_selfloop():
  stormvogel.to_dot.plot_model_pydot(pkydie_e)
  elim.eliminate_selfloop(pkydie_e, state)
 stormvogel.to_dot.plot_model_pydot(pkydie_e, highlight_state=state)  
 elim.eliminate_state(pkydie_e, state, remove=True) 
stormvogel.to_dot.plot_model_pydot(pkydie_e, highlight_state=state)
#TODO generate a carrousel with glue.
```
````

## Computing solution functions for pMDPs
Solution functions for MDPs are harder to compute due to the maximum over the policies.
In principle, they can be computed by enumerating over all memoryless, deterministic policies and for each induced MC computing the solution function.
Some alternatives have been proposed in the literature that encode a pMDP into a pMC by replacing action choices with parameters. 
The details of such approaches are beyond the scope of these notes.


# Parameter Synthesis Questions
While solution functions may be seen as the natural generalisation to values, computing them is often prohibitively expensive.
This motivates the study of problems and algorithms that do not take the solution function as intermediate solution.
We give an overview and first focus on the most common setup in a probabilistic model checking context, before mentioning further problems.
All problem statements here refer to reachability probabilities,  but they can be lifted to expected rewards or other temporal properties.

## Feasible parameters and verified parametric models
The key questions in parametric models are the existence and absence of parameters such that the induced models satisfy a property.
### Feasibility problems
Feasibility problems ask for the existence of parameters. 
Classical examples for feasibility problems are the existence of hyperparameters in, e.g., randomized algorithms and distributed protocols.
Like for standard MDPs, we can distinguish between the intent of the policy. 
For parametric Markov chains, these problems coincide --- we simply talk about the feasibility problem. 
```{admonition} Problem: Angelic feasibility  
Given a pMDP $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$, decide whether

$$
\exists \pmdp \in \generator{\pmdp}{R}. \exists \pi \in \Policies. \pr^\pi_\pmdp(\lozenge T) \bowtie \lambda
$$

```
```{admonition} Problem: Demonic feasibility  
Given a pMDP $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$, decide whether

$$
\exists \pmdp \in \generator{\pmdp}{R}. \forall \pi \in \Policies. \pr^\pi_\pmdp(\lozenge T) \bowtie \lambda
$$

```
Specifically, in angelic feasibility, we assume that the policy is helping to achieve the property. 
A few remarks are in order:
- Angelic feasibility can be reformulated into a pMC problem via [this construction](#sec:pmc:representingpolicies).
- Natural variations to the problems above are the optimization variants for all these problems.
- It suffices to consider memoryless deterministic policies.

### Verification problems
We can consider the duals of the feasibility problems which intend to verify
that for every MDP, a property holds. Verification problems are often studied to ensure robustness against variations in the probability distributions.
```{admonition} Problem: Angelic verification  
Given a pMDP $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$, decide whether

$$
\forall \pmdp \in \generator{\pmdp}{R}. \exists \pi \in \Policies. \pr^\pi_\pmdp(\lozenge T) \bowtie \lambda.
$$

```
```{admonition} Problem: Demonic verification  
Given a pMDP $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$, decide whether

$$
\forall \pmdp \in \generator{\pmdp}{R}. \forall \pi \in \Policies. \pr^\pi_\pmdp(\lozenge T) \bowtie \lambda.
$$

```
Some remarks again:
- The dual to angelic (demonic) feasibility is demonic (angelic) verification, respectively. 
- Demonic verification can be formulated over pMCs with a lot of additional parameters, just like angelic feasibility above. 
- It suffices to consider memoryless deterministic policies.


### Partitioning
In the problem statements above, the region of valuations is fixed. 
A classical problem in the analysis of parametric MDPs is to find the set of valuations that induce an MDP that satisfies the property.
```{prf:definition} Safe region
Given a pMDP $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$, the _safe region_ is
$$R_{\mathsf{safe}} = \{ \val  \in R \mid \pr^\pi_\pmdp[\val](\lozenge T) \bowtie \lambda \}, $$
and the unsafe region is $R_{\mathsf{unsafe}} = R \setminus R_{\mathsf{safe}}$
```
To check whether any individual region is safe (or unsafe), i.e., whether $R = R_{\mathsf{safe}}$, is a standard parametric verification problem, see above.
```{admonition} Problem: Exact partitioning 
Given a pMDP $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$, 
compute $R_{\mathsf{safe}}$.
```
However, the exact representation of this set is often infeasible. 
In fact, the following lemma clarifies the shape of an exact solution to the partitioning problem:
```{prf:lemma} Characterisation of safe region
:label:lem:par:saferegionfunction
Given a pMDP $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$
Let $\solfunc$ be the solution function on $R$ for reaching $T$. Then:
$$R_{\mathsf{safe}} = \{ \val \mid \solfunc[val] \bowtie \lambda \}.$$
```
````{prf:example}
:label:ex:par:exactpartition
```{code-cell} python
:tags: [remove-input]
threshold = Fraction(1,6)   
target_label = "rolled3"         
prop = f"P=? [F \"{target_label}\"]"
```
The following example continues with  @ex:parametric:kydie and gives the safe region for the reachability probability to {eval}`target_label` over {eval}`threshold`.
```{code-cell} python
:tags: [remove-input]
from stormvogel.parametric.region import plot_regions  
solfunc=stormvogel.model_checking(pkydie, prop).at_init()
_ = plot_regions([], threshold=threshold, solution_fn=solfunc, shade_safe=True, param_order=["x","y"], x_lim=(0.0001,0.99999), y_lim=(0.0001,0.99999))  
```
In fact, the boundary is given by the solution function: {eval}`str(solfunc-threshold)`.
````
In particular, given the continuity of the solution function on the well-defined region, the expression $\solfunc -\lambda = 0$ describes exactly the boundary between the safe and unsafe region!
To avoid the limitations due to the size of the solution function, a popular approach is to approximate this set.
```{admonition} Problem: Approximate partitioning 
Given a pMDP $\pmdp$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$,
compute sequences ${R_{\mathsf{safe}}}_{,i}$ and ${R_{\mathsf{unsafe}}}_{,i}$ such that:
-  ${R_{\mathsf{safe}}}_{,i} \subset R_{\mathsf{safe}}$ for all $i$,
-  ${R_{\mathsf{unsafe}}}_{,i} \subset R_{\mathsf{unsafe}}$ for all $i$,
- $\lim_{i \rightarrow \infty} {R_{\mathsf{safe}}}_{,i} = R_{\mathsf{safe}}$,
- $\lim_{i \rightarrow \infty} {R_{\mathsf{unsafe}}}_{,i} = R_{\mathsf{unsafe}}$.
```
The notions of limits here are well-defined on mild assumptions on the shapes of the individual regions.

````{prf:example}
We now approximate the safe region from @ex:par:exactpartition. The following figures shows a coarse and a finer approximation. We also plot the boundary of the safe region for convenience.
$R_{\mathsf{safe}}$ and $R_{\mathsf{unsafe}}$ are  both represented by a union of finitely many rectangular regions.
```{code-cell} python 
:tags: [remove-input]
from stormvogel.teaching.parametric import parameter_space_partitioning                                                                                                                                      
annotated = parameter_space_partitioning(
    pkydie, prop, threshold, max_iterations=100
)                                                                                                                                                                    
_ = plot_regions([r for r in annotated if r.classify(threshold) in ["safe", "unsafe"]], threshold=threshold, solution_fn=solfunc)                                                                                                                                                                                                                                                               
annotated = parameter_space_partitioning(
    pkydie, prop, threshold, max_iterations=300
)                                                                                                                                                                    
_ = plot_regions([r for r in annotated if r.classify(threshold) in ["safe", "unsafe"]], threshold=threshold, solution_fn=solfunc)                                                                                                 
```
````
As the example already indicates, the common approach to compute such approximations is by splitting the region $R$ into smaller regions and using a verification algorithm on these smaller regions.

## Beyond standard questions
The literature considers other variations, e.g., whether the reachability probability is monotonically increasing in a parameter. 

### Robust policies
Above, we have always quantified first over the parameters / the generated MDPs and only then over the policies.
That is, the policies can be different in different MDPs.
For many planning problems, one wants to find one policy that can reach a target in every environment.

```{admonition} Problem: Robust policies   
Given a pMDP $\pmdp$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$, decide whether
$$
\exists \pi \in \Policies. \exists \pmdp \in \generator{\pmdp}{R}.  \pr^\pi_\pmdp(\lozenge T) \bowtie \lambda.
$$
```
In interval MDPs, computing robust policies is equally hard as the verification problem.
However, in parametric MDPs, finding robust policies is much harder - the policies typically require infinite amounts of memory and
even when restricting oneself to the memoryless policies with a single parameter is NP-hard.
Robust policies are currently not in the scope of these notes.

### Distributions over parameters
Furthermore, while in these notes, we assume existential or universal quantification over the parameters, 
 research also assumes a distribution (a prior) over parameter values and can ask whether a randomly generated MDP (according to the prior distribution) will satisfy a specification. 

# Algorithms for feasibility and verification
We have already discussed how exact partitioning relates to the solution function and how approximate partitioning is typically realised by
applying verification algorithms on subregions. 
We now focus on algorithms to solve feasibility and verification problems on pMCs.
Many ideas carry over to pMDPs, but they are not discussed here.
Furthermore, feasibility and verification are duals of each other, thus a complete approach for feasibility is also a complete approach for verification. 
However, positive answers on feasibility are significantly easier to obtain, and leads to different algorithmic choices.

## Encoding into the existential theory of the reals 

### Lifting the equation system
The first idea is to lift the linear equation system (with variables for the reachability probability from a state and constant transition probabilities as coefficients) 
to a nonlinear equation system where the parameters are multiplied by the variables for the states.

As a precondition to lifting these encodings, recall that we must precompute the zero states. 
We use the following lemma that allows us to compute the zero states for all graph-preserving valuations by picking the induced pMC for any fixed graph-preserving valuation.
```{prf:lemma}
Given a pMC $\pmdp$ with graph-preserving region $R$. For two valuations $\val_1, \val_2 \in R$, it holds that:
$$ \{ s \mid \pr^s_{\pmdp[\val_1]}(\lozenge T) = 0 \} = \{ s \mid \pr^s_{\pmdp[\val_2]}(\lozenge T) = 0 \}. $$
We can therefore uniquely define $\Szerogp$ as the set of zero states for any graph-preserving valuation. 
```
This allows us to state the correctness of a lifted encoding.
```{prf:theorem}
Given a pMC $\pmdp$ with states $s_0, \dots s_n$, parameters $\vec{x}$, initial state $\sinit$ and targets $T$, a graph-preserving region $R$, a _threshold_ $\lambda \in \mathbb{Q}$.
The statement 
\begin{align}
\exists \vec{x} \exists p_{s_0} \dots p_{s_n} & & & \\
& \bigland_{x \in \vec{x}} x \in R(x) & & \land \\
& \bigland_{s \in \Szerogp} p_{s} = 0 & & \land \\ 
& \bigland_{s \in T} p_{s} = 1 & & \land \\ 
& \bigland_{s \in S \setminus (T \cup \Szerogp) } p_{s} = \sum_{s'} \delta(s,s') \cdot p_{s'} & & \land \\ 
& p_{\sinit} \geq \lambda
\end{align}
is valid iff there is a feasible parameter instantiation.
```
The proof simply observes that for any assignment to the parameters, this correctly provides the standard linear equation system.
````{prf:example}
```{code-cell} python
:tags: [remove-input]
from stormvogel.teaching.parametric import feasibility_problem
from stormvogel.parametric.region import RectangularRegion
region = RectangularRegion({"x": [Fraction(1,100), Fraction(99,100)], "y": [Fraction(1,100), Fraction(99,100)]})
feasibility_problem(pkydie, target_label, threshold, region)
```
````
The statement above is a statement in the existential theory of the reals (ETR), 
which allows addition and multiplication of real-valued variables, along with inequalities and equality thereof, and any Boolean combinations of such (in)equalities.
The validity of statements in the existential theory of the reals is decidable in the complexity class $\exists\mathbb{R}$, which is contained in PSPACE. The class is also NP-hard.
Practically, these problems can be solved by computer algebra systems, or by SMT-solver of the quantifier-free fragment of nonlinear arithmetic (like z3 and cvc5).
However, the practical performance of such approaches is limited to few variables. 
As the variables in these encodings are the sum of parameters and states, this in particular also limits the amount of states that can efficiently be supported.

```{prf:remark}
The theorem and the encoding can be generalized to any well-defined valuation
 by encoding the set of zero states into the statement. 
```

### Using the solution function
Therefore, it is attractive to not use a variable for every state. In fact, we can simply use the solution function and an ETR solver as for the [safe region characterisation](#lem:par:saferegionfunction):
```{prf:lemma}
Given a pMC $\pmdp$ with targets $T$, a well-defined region $R$, a _threshold_ $\lambda \in \mathbb{Q}$,  and $\bowtie \in \{\leq,\geq\}$
Let $\solfunc$ be the solution function on $R$ for reaching $T$. Then:
The statement 
$$ \exists x \in X \solfunc \geq \lambda $$
is valid iff there is a feasible parameter instantiation.
```
Practically, the downsides of this approach are twofold: 
First, one needs to compute the solution function first, which is often quite expensive.
Second, while the number of variables is now equal to the number of parameters, all structure within the problem is lost to the solver.

### Hardness
The limited performance of solvers for ETR makes it natural to ask whether we can maybe hope to do better.
However, the feasibility problem is ETR-complete, i.e., ETR solvers are in some sense adequate tools, just like LP solvers are adequate for MDPs.


## Good guessing
To answer _yes_ to the feasibility problem (or to find a feasible valuation),
one simply needs to guess a good solution. 
Then, one can verify that this solution is indeed valid efficiently using standard model checking. 
Practically relevant feasibility solvers therefore rely on such guesses.
Different ideas have been used, often based on linearizations of the nonlinear equation system or on gradient descent.

## Parameter lifting
Let us turn our attention to a complete approach for feasibility/verification. 
Rather than relying on the ETR encodings, we will use an abstraction-refinement procedure akin to branch-and-bound methods.
```{warning} Assumptions
 For exposition, we assume simplex pMDPs with MaxReachProps and rectangular graph-preserving regions.
```
For that, we are interested in the value range of a region:
$$ 
V_{\pmdp,R} = [\min_{\mdp \in \generator{\pmdp}{R}} \pr^{\max}_{\mdp}(\lozenge T), \max_{\mdp \in \generator{\pmdp}{R}} \pr^{\max}_{\mdp}(\lozenge T) ].$$
Computing this value range for parametric models is intuitively complicated because due to parameter dependencies, we cannot reason locally, as already pointed out in @ex:parametric:kydie.
On the other hand, in absence of such dependencies, reasoning becomes much simpler.
Thus, instead of the pMDP $\pmdp$, we analyze $\abst{\pmdp}{R}$
The value range of this iMDP  is 
$$
V_\imdp = [\min_{\mdp \in \generatorint{\abst{\pmdp}{R}}} \pr^{\max}_{\mdp}(\lozenge T), 
\max_{\mdp \in \generatorint{\abst{\pmdp}{R}}} \pr^{\max}_{\mdp}(\lozenge T) ].
$$
Due to @lem:par:lifting, $V_{\pmdp,R} \subseteq V_{\abst{\pmdp}{R}}$.


Now, to solve the verification problem, we must prove that $V_\pmdp(R) \geq \lambda$.
If we compare $V_{\abst{\pmdp}{R}}$ with $\lambda$, there are 3 cases:
1. $V_{\abst{\pmdp}{R}} \geq \lambda$, then also $V_{\pmdp,R} \geq \lambda$ and we have proven the property.
2. $V_{\abst{\pmdp}{R}} \leq \lambda$, then also $V_{\pmdp,R} \leq \lambda$ and we have refuted the property.
3. $\lambda \in V_{\abst{\pmdp}{R}}$. In this case, we cannot conclude anything. 

To handle the third case, we split the region, as justified by the following observation:
```{prf:lemma}
Let $R_1, R_2 \subseteq R$ be regions with $R_1 \cup R_2 = R$. 
$$ V_{\pmdp,R} \geq \lambda \quad\text{ iff }\quad V_{\pmdp,R_1} \geq \lambda \text{ and } V_{\pmdp,R_2} \geq \lambda. $$
```
Under mild assumptions, 
the interval MDP abstraction is less coarse for smaller regions. 
In particular, due to the smoothness of the solution function on the graph-preserving region,
if we converge towards point regions, we also converge against point intervals for both the value of a region of a PMDP and the value of its abstraction.

````{prf:example}
We reconsider the following example on region $R=[0.1, 0.9]$. 
We show the situation after a few iterations and after more iterations.
On the x-axis, you see the different regions. The range of the lifted iMC is visualised,
the color indicates which of the three cases above applies for this value range, in comparison with the threshold. 
```{code-cell} python      
:tags: [remove-input]
from stormvogel.parametric.region import plot_annotated_regions_1d                                                                                                                                                                                                                                                                    
pmc = sv.model.new_dtmc(create_initial_state=False)                                                                                                                        
x = pmc.declare_parameter("x")                                                                                                                                       
                                                                                                                                                                     
s0 = pmc.new_state(["init"], friendly_name="s0")                                                                                                                     
s1 = pmc.new_state(friendly_name="s1")  
s2 = pmc.new_state(friendly_name="s2")                                                                                                                               
s3 = pmc.new_state(["T"], friendly_name="s3")                                                                                                                               
s4 = pmc.new_state(["sink"], friendly_name="s4")
                                                                                                                                                                     
pmc.set_choices(s0, [(x,     s1), (1 - x, s4)])                                                                                                                      
pmc.set_choices(s1, [(1 - x, s2), (x,     s4)])
pmc.set_choices(s2, [(x,     s3), (1 - x, s4)])                                                                                                                      
pmc.set_choices(s3, [(1,     s3)])                                                                                                                      
pmc.set_choices(s4, [(1,     s4)])                                                                                                                      

stormvogel.to_dot.plot_model_pydot(pmc)
threshold = 0.12             
solfunc = stormvogel.model_checking(pmc, "P=? [F \"T\"]").at_init()                                                                                                                     
annotated = parameter_space_partitioning(
    pmc, f"P<=0.2 [F \"T\"]", threshold, initial_region=RectangularRegion({"x": [Fraction(1,10), Fraction(9,10)]}), max_iterations=5
)  
_ = plot_annotated_regions_1d(annotated, threshold, solution_fn=solfunc)
annotated = parameter_space_partitioning(
    pmc, f"P<=0.2 [F \"T\"]", threshold, initial_region=RectangularRegion({"x": [Fraction(1,10), Fraction(9,10)]}), max_iterations=20
)  
_ = plot_annotated_regions_1d(annotated, threshold, solution_fn=solfunc)
```
````