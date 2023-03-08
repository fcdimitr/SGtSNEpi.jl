<div align="center">
    <img src="https://raw.githubusercontent.com/fcdimitr/SGtSNEpi.jl/master/docs/src/assets/logo.png" alt="SGtSNEpi.jl" width="800">
</div>

<br/>

[![Build Status](https://github.com/fcdimitr/SGtSNEpi.jl/workflows/CI/badge.svg)](https://github.com/fcdimitr/SGtSNEpi.jl/actions)
[![Coverage](https://codecov.io/gh/fcdimitr/SGtSNEpi.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/fcdimitr/SGtSNEpi.jl)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fcdimitr.github.io/SGtSNEpi.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fcdimitr.github.io/SGtSNEpi.jl/dev)

We provide a `Julia` interface, i.e., a wrapper to [SG-t-SNE-Π](http://t-sne-pi.cs.duke.edu), which
is a high-performance software for swift embedding of a large, sparse
graph into a d-dimensional space (d = 1,2,3) on a shared-memory
computer.

## Installation

To install `SG-t-SNE-Π` through Julia, issue

```julia
] add SGtSNEpi
```

> :warning: **SGtSNEpi is currently not working on Windows and native M1 Macs**: Either use WSL2 on Windows or use the package via rosetta2 on M1 Macs.

See [the full
documentation](https://fcdimitr.github.io/SGtSNEpi.jl/stable) for more
details.

## Citation

If you use this software, please cite the following paper:

```bibtex
@inproceedings{pitsianis2019sgtsnepi,
    author = {Pitsianis, Nikos and Iliopoulos, Alexandros-Stavros and Floros, Dimitris and Sun, Xiaobai},
    doi = {10.1109/HPEC.2019.8916505},
    booktitle = {IEEE High Performance Extreme Computing Conference},
    month = {11},
    title = {{Spaceland Embedding of Sparse Stochastic Graphs}},
    year = {2019}
}
```