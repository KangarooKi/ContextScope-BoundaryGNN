# Research Idea: Audience Boundary Risk in Ego Networks

## 5W1H

**What**: Predict which friends in an ego network are likely to bridge otherwise
separate audience contexts.

**Why**: Social platforms expose users to context collapse: family, classmates,
coworkers, and hobby communities may all see or indirectly discuss the same post.
Standard graph tasks rarely ask whether a recipient is risky for a specific
audience boundary.

**Who**: Social computing researchers, privacy-tool designers, recommender-system
teams, and users who curate friend lists.

**When**: At post-composition time, before choosing a friend list or audience group.

**Where**: Ego-network platforms with friend lists, circles, group membership, or
audience controls.

**How**: Use public SNAP Google+ circles as weak supervision. Derive a boundary
risk label from multi-circle membership and cross-circle neighborhoods. Compare
three baselines with a relation-aware GNN that separates homophilous edges from
dissimilar bridge edges.

## Task Definition

For every friend node in an ego network, predict:

```text
boundary_risk(friend) = 1
```

when the friend has evidence of connecting multiple audience contexts. The current
implementation uses two signals from public circle labels:

1. membership in more than one circle;
2. a high ratio of neighbors whose circle memberships fall outside the friend's
   own circle set.

Circle labels are used only to derive supervision and evaluate the task. Model
features come from anonymized profile attributes and graph structure.

## Innovation Angle

The project is not another social circle detector. It treats existing circles as
user-authored audience boundaries and asks which nodes make those boundaries weak.
This turns the dataset into a privacy and interface-assistance benchmark, with
counterfactual-style review suggestions after prediction.

## Compared Methods

The default experiment compares only four methods:

- `ProfileLogit`: profile-only baseline.
- `StructureLogit`: graph-structure baseline.
- `ProfileStructureLogit`: non-message-passing profile plus structure baseline.
- `BoundaryGNN-Logit`: our method, combining ordinary neighbor aggregation with
  separate similar-edge and dissimilar-edge aggregation channels.
