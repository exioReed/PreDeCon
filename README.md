# PreDeCon

This repository is not associated with the original authors of [Boehm,2004].

## About

_Subspace Preference Weighted Density Connected Clustering_ (PreDeCon) [Boehm,2004] can be seen as a
modification to the famous DBSCAN [Ester,1996] that addresses problems which arise in
high-dimensional spaces.

## Installation

Install with `pip`.

From PyPI

```
$ pip install predecon-exioreed
```

Alternatively, from source

```
$ pip install git+https://github.com/exioReed/PreDeCon@master#egg=PreDeCon-exioreed
```

or

```
$ git clone https://github.com/exioReed/PreDeCon.git
$ cd PreDeCon
$ pip install .
```

## References

`[Boehm,2004]` Boehm, C. et al., "Density Connected Clustering with Local Subspace Preferences".
In: _Proceedings of the 4th IEEE Internation Conference on Data Mining (ICDM)_,
Brighton, UK, 2004.

`[Ester,1996]` Ester, M. et al., "A Density-Based Algorithm for Discovering Clusters in Large
Spatial Databases with Noise".
In: _Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining_,
Portland, OR, 1996.