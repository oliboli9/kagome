# Kagome Weaving
This project is part of a Bachelor Thesis titled "Kagome Weaving in Architectural Design: a Mathematical Optimisation Problem". It provides a framework for generating feasible Kagome weave patterns for various topologies. 

## Examples and Extensions
Three examples can be found as demos for a capped cylinder, a torus, and a periodic sinusoidal surface. The periodic surface demo is set up for parallel execution of the same surface with different weave densities. The `n_processes` variable should be adjusted according to the number of cores available. 

All necessary tools can be found under `methods/`. New surfaces can be defined following the `Surface` interface in `methods/surface.py`. 

### Contact
In case of questions, please feel free to contact the author at s215142@dtu.dk