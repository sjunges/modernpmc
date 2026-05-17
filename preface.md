---
kernelspec:
  name: python3
  display_name: Python 3
---

# Preface
These lecture notes are written for the course Model Checking at Radboud University and cover part of the material.
The notes are currently in alpha state.

```{attention}
With the attention blocks, we highlight blocks that are still missing. 
Various examples are still missing and most citations are still missing.
```


## Thanks
A lot of material is inspired on chapter ten of the book Principles of Model Checking by Christel Baier and Joost-Pieter Katoen, 
 on the second edition of the book Markov Decision Processes by Martin Puterman, and on lecture slides by Dave Parker.
I am thankful for the material shared by Tim Quatmann and Joost-Pieter Katoen from RWTH Aachen University, especially on multiobjective model checking, as well as Material from Nils Jansen and Maximilian Weininger from Ruhr University Bochum.
I am grateful for the comments by Eline Bovy on an earlier version of these lecture notes.
I want to thank Matthias Volk for improvements in CI and PDF building.

## Code examples
Examples in these lecture notes are generated using stormvogel and the stormvogel teaching module. 
We compute various results using the [storm model checker](www.stormchecker.org), via the python bindings `stormpy`. 
We generate equations (for displaying) via sympy and visualise graphs with pydot.
In particular, the following import statements preceed all code used in these lecture notes.
```{code-cell} python
:tags: [remove-outputs]
import stormpy
import stormpy.info
import stormvogel as sv
import stormvogel.teaching as teach
import stormvogel.bird as bird
import sympy
sympy.init_printing()
from IPython.display import Math
```
The version used in these notes is stormpy {eval}`stormpy.info.storm_version()`.
We currently run a custom version of stormvogel with the intent of merging this back. 

To not disrupt the reading flow to much, I've decided to mostly not show the code generating the results.
However, the markdown can be downloaded for every chapter.

## Feedback
I would be very happy to receive feedback. 
Every section has a discuss button (available when hovering over the section title) which brings you straight to a [github discussion page](https://github.com/sjunges/modernpmc/discussions). 
Feedback can also be posted by mail.

## License
This book is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).