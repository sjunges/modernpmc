
### What are tractable hyperproperties?
Hyperproperties, in a nutshell, allow comparing different paths and path properties with eachother.
While in general, hyperproperties are very powerful and most questions about them are undecidable in the probabilistic setting, 
we consider the smaller fragment of so-called tractable hyperproperties, in particular we consider relational reachability.
With relational reachability, we can compare different reachability probabilities.

```{prf:example} (Modelling unbiased coins with interval-biased coin flips, from @)
Say we want simulate an unbiased, perfectly random coin flip. 
We only have access to an old coin that most likely has some bias, and we can flip this  coin as much as we want.
That is, we have access to an  
 using an infinite stream of \emph{biased} random bits,
each of which is $0$ with some unknown but fixed probability $0<p<1$.
A simple solution @vonneumannVariousTechniques1951 : 
Extract the first two bits from the stream; if they are different, return the value of the first; otherwise try again.
Checking that this method is correct is shown in @ex:pmc:vonNeumannTrick.

%
Now, consider a variation of the problem where the stream comprises random bits with 
\emph{different}, unknown biases $p_0, p_1, {\ldots}$ which are, however, all known to lie in an 
interval $[\underline{p}, \overline{p}] \subset (0,1)$~\cite{gerlachEfficientProbabilistic2025}.
Is von Neumann's solution still applicable in this new situation?
%
To address this question for a concrete interval $[\underline{p}, \overline{p}]$, say $[0.59,0.61]$, 
we may model the situation as shown in~\Cref{fig:illustrativeExample} and formalize the 
corresponding relational reachability property as:
%
\begin{align}
    \forall \sched .\
    \Pr_{s_0}^{\sched}(\Finally \{01\}) \,\approx_{\epsilon}\, \Pr_{s_0}^{\sched}(\Finally \{10\})
    \tag{$\dagger$}\label{eq:vonNeumannExampleProperty}
\end{align}
%
where $\sched$ is a universally quantified scheduler, 
$s_0$ is the initial state, and $\approx_{\epsilon}$ means approximate equality up to an absolute error of $\epsilon \geq 0$.
%
Note that the universal quantification in \eqref{eq:vonNeumannExampleProperty} is over \emph{general} schedulers that may use both unbounded memory and randomization.
This is essential to model the problem properly:
Without randomization, all biases would be either $\underline{p}$ or $\overline{p}$ and a bounded-memory scheduler would induce an ultimately periodic stream of biases.

```




### How to solve relational reachability properties?  
We consider only the simplest properties here, i.e.


Using the techniques presented in this paper, we can establish automatically that, as expected, von Neumann's trick continues to work \enquote{approximately} in the new setting.
More precisely, in~\cref{sec:eval_new} we use our algorithm to automatically verify that \eqref{eq:vonNeumannExampleProperty} is false for $\epsilon = 0$ (exact equality), but holds if we relax the constraint to $\epsilon = 0.1$. 
}%
%
Our algorithm proceeds as follows~(see \cref{sec:reach_algo}):
First, we unfold the MDP w.r.t.\ the target states $01$ and $10$, as depicted in~\cref{fig:illustrativeExampleUnfolded}.
Then, we define reward structures $\rew_{01}, \rew_{10}$ on the unfolded MDP $\mdp'$, collecting reward 1 in states $01$ and $10$, respectively.
In order to check whether the desired property holds, we can then check whether $\forall \sched \in \Scheds[\mdp'] .\ \Expected^{\mdp',\sched}_{\state_0}(\rew_{01} - \rew_{10}) \approx_\epsilon 0$, which is equivalent to checking that $\min_{\sched \in \Scheds[\mdp']} \Expected^{\mdp',\sched}_{\state_0}(\rew_{01} - \rew_{10}) \geq - \epsilon \land \max_{\sched \in \Scheds[\mdp']} \Expected^{\mdp',\sched}_{\state_0}(\rew_{01} - \rew_{10}) \leq \epsilon$.
