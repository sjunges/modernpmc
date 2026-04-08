---
kernelspec:
  name: python3
  display_name: Python 3
---

# Preface
These lecture notes are written for the course Model Checking at Radboud University and cover part of the material.
The notes are currently in alpha state.

## Thanks
I am thankful for the material shared by Tim Quatmann and Joost-Pieter Katoen from RWTH Aachen University, as well as from Nils Jansen and Maximilian Weininger from Ruhr University Bochum.
Furthermore, the material is partially based on the book Principles of Model Checking by Christel Baier and Joost-Pieter Katoen 
and on the second edition of the book Markov Decision Processes by Martin Puterman.

## Code examples
Examples in these lecture notes are generated using stormvogel and the stormvogel teaching module. 
We compute various results using the storm model checker, via the python bindings. 
We generate equations (for displaying) via sympy.
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
import stormvogel.teaching.bellman as bellman
```
The version used in these notes is stormpy {eval}`stormpy.info.storm_version()`.

To not disrupt the reading flow to much, I've decided to mostly not show the code generating the results.
However, the markdown can be downloaded for every chapter.

## License
This book is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).