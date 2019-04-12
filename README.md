# machine-learning-applied-to-cfd

## Introduction

This repository contains examples of how to use machine learning (ML) algorithms
in the field of computational fluid dynamics (CFD). ML algorithms may be applied in different steps during a CFD simulation:

- **pre-processing**, e.g. for geometry or mesh generation
- **run-time**, e.g. as a dynamic boundary condition or as a subgrid-scale model
- **post-processing**, e.g. to create substitute models or to analyze results

Another possible categorization is to distingish the type of learning, e.g.

- **supervised learning:** the algorithm creates a mapping between given features and labels, e.g. between the shape of a truck and the drag force acting on it
- **unsupervised learning:** the algorithm finds labels in the data, e.g. if two particles *p1* and *p2* are represented by some points on their surface (there is only a list of points but it is not known to which particle they belong), the algorithm will figure out for each point weather is belongs to *p1* or *p2*
- **reinforcement learning:** an agent acting in an environment tries to maximize a (cumulative) reward, e.g. an agent setting the solution control of a simulation tries to finish the simulation as quickly as possible, thereby learning to find optimized solution controls for a given set-up (*agent*: some program modifying the solver settings; *environment*: the solver reacting on the changes in the settings; *reward*: the inverse of the time required to complete one iteration)

## How to reference

If you found useful examples in this repository which helped you to write a scientific publication, you may consider citing one or more of the following references:

```
@article{doi:10.1002/ceat.201900044,
author = {Weiner, Andre and Hillenbrand, Dennis and Marschall, Holger and Bothe, Dieter},
title = {Data-driven subgrid-scale modeling for convection-dominated concentration boundary layers},
journal = {Chemical Engineering \& Technology},
}
```

## Examples grouped by the type of learning

### Supervised learning

- **regression:** approximating the shape of a bubble
  - 2D, steady, ellipsoidal
  - 2D, steady, skirted
  - 2D, steady, ellipsoidal, varying equivalent diameter
  - 3D, unsteady, ellipsoidal
  - 3D, unsteady, ellipsoidal, varying equivalent diameter
- **regression:** approximating the transient rise velocity of a bubble
  - single bubble
  - varying equivalent diameter
- **classification:** transition between various paths regimes of rising bubbles
  - binary classification
  - multi-class classification

### Unsupervised learning

- **clustering:** finding small artefacts or particles in a Volume-of-Fluid simulation
- **clustering:** counting the number of liquid drops in atomization

### Reinforcement learning

- accelerating iterative solvers

## Other useful examples

## Journal articles realted to ML + CFD

- [Data‐driven subgrid‐scale modeling for convection‐dominated concentration boundary layers (2019)](https://onlinelibrary.wiley.com/doi/abs/10.1002/ceat.201900044)
- [Turbulence Modeling in the Age of Data (2019)](https://www.annualreviews.org/doi/abs/10.1146/annurev-fluid-010518-040547)
- [Deep learning in fluid dynamics (2017)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/deep-learning-in-fluid-dynamics/F2EDDAB89563DE5157FC4B8342AD9C70)

## Contributors

- [Andre Weiner](https://github.com/AndreWeiner), [Mail](weiner@mma.tu-darmstadt.de)

## Other repositories with related content

- [Computational-Fluid-Dynamics-Machine-Learning-Examples](https://github.com/loliverhennigh/Computational-Fluid-Dynamics-Machine-Learning-Examples)
