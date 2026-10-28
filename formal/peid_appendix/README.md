# Scoped Lean verification of the PEID appendix

This directory checks hierarchical additivity, algebraic information
identities and conditional two-source PID arguments against the appendix
snapshot exported on 2026-10-27.

Use Lean **4.29.0**, as pinned by `lean-toolchain`. Mathlib is not required.
From the repository root, run:

```sh
python formal/peid_appendix/verify.py
```

The verifier compiles `Appendix.lean` and `Counterexamples.lean`, inspects the
axiom dependencies of the named theorems and writes `verification.log` and
`verification.json`. Completed proofs must use Lean's standard logical axioms;
unproved placeholders and custom axioms are rejected. The counterexample file
also examines the spectral-ordering sign claim in the source snapshot.

## Scope

Information quantities are represented as values in abstract ordered rings or
fields. Probabilistic conclusions assume the entropy identities, independence
consequences, data processing and nonnegativity hypotheses explicitly stated in
the declarations. Shannon entropy, KL divergence and conditional probability
have not been constructed from their foundational definitions here. Passing
these checks establishes the stated conditional arguments, rather than a full
formalization of all probability theory or of the entire appendix.

`source/provenance.json` records source checksums and manuscript provenance.
`source/appendix_S1_S4_20261027.tex` retains the S1-S4 source excerpt used for
this verification; full-line comments were removed. The compiled-text snapshot
in the same directory records the equation numbering. These snapshots describe
the verification target and may differ from later manuscript revisions.
