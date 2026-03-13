
# Appendix

## Distributions

A **discrete distribution** over a set $X$ is a function
$
\mu\colon X \rightarrow [0,1]
$
such that
$
\sum_{x \in X} \mu(x) = 1 .
$
The **support** of $\mu$ is
$
\supp{\mu} = \{ x \mid \mu(x) > 0 \}.
$

A distribution $\mu$ is Dirac, if $|\supp{\mu}|=1$.

## Fixpoint theory

### Lattices
Let $(L, \leq)$ be a partially ordered set, and let $X \subseteq L$.
- An element $u \in L$ is an _upper bound_ of $X$ if $x \leq u$ for all $x \in X$. 
- An element $l \in L$ is a _lower bound_ of $X$ if $l \leq x$ for all $x \in X$.
- The _least upper bound_ (or _join_) of $X$, denoted $\bigvee X$, is an upper bound $u$ of $X$ such that for any other upper bound $u'$ of $X$, $u \leq u'$.  
- The _greatest lower bound_ (or _meet_) of $X$, denoted $\bigwedge X$, is a lower bound $l$ of $X$ such that for any other lower bound $l'$ of $X$, $l' \leq l$.

```{prf:definition} (complete) latttices
A _lattice_ $(L, \leq)$ is a partially ordered set in where each pair $x, y \in L$ has a join and a meet in $L$.  
A lattice is  _complete_, if every subset $X \subseteq L$ has a join $\bigvee X$ and a meet $\bigwedge X$ in $L$.
```

### Monotone and Continuous Operators

Let $(L, \leq)$ be a partially ordered set. 

```{prf:definition} monotone operator
An operator $F\colon L \to L$ is _monotone_ if for all $x, y \in L$:
$$
x \leq y \implies F(x) \leq F(y) 
$$
```
```{prf:definition} ω-continuous operator
Let $(L, \leq)$ be a complete lattice and $F\colon L \to L$ a monotone operator.  
The operator $F$ is _ω-continuous_ if it preserves suprema of countable ascending chains. Formally:
Let $(x_i)_{i \ge 0}$ be a countable ascending chain in $L$, i.e., 
$$
x_0 \leq x_1 \leq x_2 \leq \dots
$$
then $F$ is ω-continuous if:
$$
F\Big(\bigvee_{i \ge 0} x_i\Big) = \bigvee_{i \ge 0} F(x_i)
$$
```
The intuition here is that applying $F$ to the “limit” of an ascending chain is the same as taking the limit of applying $F$ to each element. 
This property is crucial for infinite lattices, which we encounter in analysing MDPs.

### Fixpoints

```{prf:definition} Fixpoint
An element $x \in L$ is a _fixpoint_ of $F\colon L \to L$ if

$
F(x) = x
$
```
The set of all fixpoints of $F$ is denoted $\mathrm{Fix}(F)$.

```{prf:theorem} Knaster-Tarski Theorem
Let $(L, \leq)$ be a complete lattice** and $F\colon L \to L$ be monotone. Then:

1. $F$ has a _least fixpoint_ $\lfp(F)$ and a _greatest fixpoint_ $\gfp(F)$.  
They are given by

$$
\lfp(F) = \bigwedge \{ x \in L \mid F(x) \leq x \}, \quad
\gfp(F) = \bigvee \{ x \in L \mid x \leq F(x) \}
$$

2. If $(x_i)_{i \ge 0}$ is defined by

$$
x_0 = \bot, \quad x_{i+1} = F(x_i)
$$

then $\lfp(F) = \bigvee_{i \ge 0} x_i$.
```

For **finite lattices**, the least fixpoint $\lfp(F)$ can be computed by iterating $F$ starting from the least element $\bot$ until a fixpoint is reached. Similarly, the greatest fixpoint $\gfp(F)$ can be obtained by iterating from the greatest element $\top$.

