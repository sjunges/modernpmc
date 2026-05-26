---
numbering:
  heading_1: true
  heading_2: true
  heading_3: true
  equations: false

---

# Appendix

## Notation

Let $b$ be a Boolean expression. $$ \indicator{b} = \begin{cases} 1 & \text{if } b \\ 0 & \text{otherwise.} \end{cases} $$
We often use the indicator for expressions such as $\indicator{x = 0}$.

## Polynomials and rational functions
Let $X = \{ x_0, \dots, x_n \}$ denote an (ordered) set of variables.
A _polynomial_ over $X$ is an expression of the shape $$ \sum_{i}^m c_i x_0^{e_{i,0}} \cdot x_1^{e_{i,1}} \cdot \dots \cdot x_n^{e_{i,n}}$$
with exponents $e_{i,j} \in \mathbb{N}$ and coefficients $c_i \in \mathbb{Q}$. We denote the set of polynomials as $\poly{X}$. 

A _rational function_ is a fraction of two polynomials. The set of rational functions is written \ratfunc{X}$.

(app:geometry)=
## Geometry

Let $\vec{p} = (p_1, \ldots, p_m) \in \mathbb{R}^m$ be a point in an $m$-dimensional Euclidean space.  
For $c \in \mathbb{R}$, let $c \cdot \vec{p} = (c \cdot p_1, \ldots, c \cdot p_m)$ be a scalar multiplication.
-  $R \subseteq \mathbb{R}^m$ is _convex_ iff $\vec{p}, \vec{q} \in R$ implies  
$c \cdot \vec{p} + (1 - c)\cdot \vec{q} \in R$ for all $c \in [0,1]$.
- $R \subseteq \mathbb{R}^m$ is _downward-closed_ iff $\vec{p} \in R$ and $\vec{p} \geq \vec{q} \in \mathbb{R}^m$ implies $\vec{q} \in R$.

We say $\vec{p} \in \mathbb{R}^m$ _dominates_ $\vec{q} \in \mathbb{R}^m$, written $\vec{p} \succ \vec{q}$, iff $\vec{p} \geq \vec{q}$ and $\vec{p} \ne \vec{q}$.

For a weight vector $\vec{w} \in \mathbb{R}^m_{\geq 0}$ and a point $\vec{p} \in \mathbb{R}^m$, the _supporting half-plane_ induced by $\vec{w}$ and $\vec{p}$ is
$$H(\vec{w}, \vec{p}) = \{ \vec{q} \in \mathbb{R}^m \mid \vec{w} \cdot \vec{q} \leq \vec{w} \cdot \vec{p} \}.$$



## Distributions

A _discrete distribution_ over a set $X$ is a function
$
\mu\colon X \rightarrow [0,1]
$
such that
$
\sum_{x \in X} \mu(x) = 1 .
$
The _support_ of $\mu$ is
$
\supp{\mu} = \{ x \mid \mu(x) > 0 \}.
$

A distribution $\mu$ is _Dirac_, if $|\supp{\mu}|=1$.

(app:dfa)=
## Deterministic finite automata

```{prf:definition}
A _deterministic finite automaton_ is a tuple 
$$
\langle Q, \Sigma, q_0, \delta, \textsf{Acc} \rangle
$$
with 
- a finite set of states $Q$,
- a finite alphabet $\Sigma$,
- an initial state $q_0 \in Q$,
- a transition relation $\delta\colon Q \times \Sigma \times Q$,
- A set of _accepting states_ $\textsf{Acc} \subseteq Q$.
```
For a given alphabet, a word $w$ is a sequence of alphabet symbols, i.e., $w \in \Sigma^{*}$. 
We use $\epsilon$ to denote the sequence of length zero.

We lift the transition relation $\delta$ inductively to words: $\delta \colon Q \times \Sigma^* \rightarrow Q$.
Intuitively, we simply follow the transitions by the DFA to arrive at a state. Formally, we inductively define this transition as follows:
- $\delta(q, \epsilon) = q$, and
- $\delta(q, w \cdot a) = \delta(\delta(q,w), a).$

```{prf:definition} DFA language
:label:def:dfa:language
The language accepted by the DFA is the set of words that end in a final state $$ \mathcal{L}(\dfa) = \{ w \in \Sigma^{*} \mid \delta(q_\text{init}, w) \in F \}$$.
```

(app:fixpoints)=
## Fixpoint theory

### Lattices
Let $(L, \leq)$ be a partially ordered set, and let $X \subseteq L$.
- An element $u \in L$ is an _upper bound_ of $X$ if $x \leq u$ for all $x \in X$. 
- An element $l \in L$ is a _lower bound_ of $X$ if $l \leq x$ for all $x \in X$.
- The _least upper bound_ (or _join_) of $X$, denoted $\bigvee X$, is an upper bound $u$ of $X$ such that for any other upper bound $u'$ of $X$, $u \leq u'$.  
- The _greatest lower bound_ (or _meet_) of $X$, denoted $\bigwedge X$, is a lower bound $l$ of $X$ such that for any other lower bound $l'$ of $X$, $l' \leq l$.

```{prf:definition} (complete) latttices
:label: def:lattice
A _lattice_ $(L, \leq)$ is a partially ordered set in where each pair $x, y \in L$ has a join and a meet in $L$.  
A lattice is  _complete_, if every subset $X \subseteq L$ has a join $\bigvee X$ and a meet $\bigwedge X$ in $L$.
```

(app:fixpointoperators)=
### Monotone and Continuous Operators

Let $(L, \leq)$ be a partially ordered set. Operators are functions from $L$ to $L$.

```{prf:definition} monotone operator
An operator $\Psi\colon L \to L$ is _monotone_ if for all $x, y \in L$:
$$
x \leq y \implies \Psi(x) \leq \Psi(y) 
$$
```
```{prf:definition} $\omega$-continuous operator
Let $(L, \leq)$ be a complete lattice and $\Psi\colon L \to L$ a monotone operator.  
The operator $F$ is _$\omega$-continuous_ if it preserves suprema of countable ascending chains. Formally:
Let $(x_i)_{i \ge 0}$ be a countable ascending chain in $L$, i.e., 
$$
x_0 \leq x_1 \leq x_2 \leq \dots
$$
then $\Psi$ is $\omega$-continuous if:
$$
\Psi\Big(\bigvee_{i \ge 0} x_i\Big) = \bigvee_{i \ge 0} \Psi(x_i)
$$
```
The intuition here is that applying $F$ to the “limit” of an ascending chain is the same as taking the limit of applying $F$ to each element. 
This property is crucial for infinite lattices, which we encounter in analysing MDPs.

### Fixpoints

```{prf:definition} Fixpoint
An element $x \in L$ is a _fixpoint_ of $\Psi\colon L \to L$ if

$
\Psi(x) = x
$
```
The set of all fixpoints of $\Psi$ is denoted $\mathrm{Fix}(\Psi)$.

```{prf:theorem} Knaster-Tarski Theorem
Let $(L, \leq)$ be a complete lattice and $\Psi\colon L \to L$ be monotone. Then:

1. $\Psi$ has a _least fixpoint_ $\lfp{\Psi}$ and a _greatest fixpoint_ $\gfp{\Psi}$.  
   They are given by
   
   $$
   \lfp{\Psi} = \bigwedge \{ x \in L \mid \Psi(x) \leq x \}, \quad
   \gfp{\Psi} = \bigvee \{ x \in L \mid x \leq \Psi(x) \}
   $$

2. If $(x_i)_{i \ge 0}$ is defined by

   $$
   x_0 = \bot, \quad x_{i+1} = \Psi(x_i)
   $$

   then $\lfp{\Psi} = \bigvee_{i \ge 0} x_i$.
```

For _finite lattices_, the least fixpoint $\lfp{\Psi}$ can be computed by iterating $\Psi$ starting from the least element $\bot$ until a fixpoint is reached.
Similarly, the greatest fixpoint $\gfp{\Psi}$ can be obtained by iterating from the greatest element $\top$.

