# Coupled Standard Map MLP+PEID Design

## Goal

Test whether an MLP can learn a noisy two-rotor coupled standard map with low
one-step prediction error and recover the Oracle PEID ranking and strength.
Run a mixed trajectory/intervention training regime only if the trajectory-only
model misses preregistered prediction or PEID thresholds.

## Dynamics

For rotor indices \(i,j\in\{1,2\}\), \(i\ne j\), define

\[
I_{i,t}=K\sin q_{i,t}+J\sin(q_{j,t}-q_{i,t})+\epsilon_{i,t},
\]

\[
p_{i,t+1}=\operatorname{wrap}(p_{i,t}+I_{i,t}),\qquad
q_{i,t+1}=\operatorname{wrap}(q_{i,t}+p_{i,t+1}).
\]

Use \(K=1.5\), \(J=0.8\), and Gaussian impulse noise with standard deviation
\(0.05\). The MLP input is the periodic encoding of
\((q_1,p_1,q_2,p_2)\), and its supervised target is \((I_1,I_2)\).

## Ground Truth

The structural hyperedges are
\(\{q_1,q_2\}\to I_1\) and \(\{q_1,q_2\}\to I_2\). Momentum variables do not
enter either impulse equation and are structural null sources. Under independent
uniform angle interventions, the squared mixed-derivative interaction has the
analytic value

\[
\mathbb E[(\partial^2 I_i/\partial q_1\partial q_2)^2]=J^2/2.
\]

Oracle PEID computed from the known stochastic map on the same intervention
samples is the numerical information-theoretic ground truth. Structural
interaction and PEID residual are reported separately.

## Data And Models

Generate independent trajectories and split by trajectory ID. The primary
Trajectory-MLP uses only trajectory states. Inputs use sine/cosine encoding for
all angular coordinates. Prediction metrics are impulse \(R^2\), impulse NRMSE,
and reconstructed next-state circular MAE.

If the primary model fails any preregistered gate, train Mixed-MLP on equal
numbers of trajectory states and independent uniform states over the full
four-dimensional torus. Targets for both regimes are sampled from the same
known stochastic transition.

## PEID Evaluation

Use shared independent uniform intervention samples for Oracle and MLP. Compute
single-source EI and all six two-source PEID residuals for both impulse targets.
The transport-map estimator is used for the full experiment; histogram PEID is
available for smoke tests. Fixed output noise prevents singular deterministic
continuous mutual information.

Trajectory-MLP passes only if:

- impulse test \(R^2\ge0.99\) and NRMSE \(\le0.05\);
- reconstructed next-state circular MAE \(\le0.05\) radians;
- \(\{q_1,q_2\}\) is the strongest source pair for both targets;
- Oracle/MLP hyperedge Spearman correlation is at least \(0.9\);
- both true-hyperedge relative errors are at most \(20\%\);
- every momentum single-source EI is at most \(0.02\) bit.

## Outputs

Create JSON/NPZ caches, one directly viewable PNG summary, and a concise
Markdown report. The figure compares prediction metrics, Oracle versus MLP
hyperedges, and true-hyperedge strengths. Legends remain outside the axes.

