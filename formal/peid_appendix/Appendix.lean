import Lean

/-!
PEID appendix verification, live Overleaf export checked 2026-10-27.

The algebraic statements quantify over arbitrary commutative rings/ordered
fields, rather than replacing information values with integers or floats.
No probability distribution, logarithm, entropy, or conditional density is
silently axiomatized here. Information-theoretic inputs to the conditional
results are explicit theorem arguments; see the accompanying coverage report.
-/

set_option autoImplicit false

namespace PEID
open Lean.Grind

section Algebra
variable {R : Type} [CommRing R]
variable {ι : Type}

/-- Finite sums use lists; a partition is represented by an enumeration of its blocks. -/
def sumOn (xs : List ι) (f : ι → R) : R :=
  match xs with
  | [] => 0
  | x :: rest => f x + sumOn rest f

theorem sumOn_append (xs ys : List ι) (f : ι → R) :
    sumOn (xs ++ ys) f = sumOn xs f + sumOn ys f := by
  induction xs with
  | nil => simp [sumOn]; grind
  | cons x xs ih => simp [sumOn, ih]; grind

theorem sumOn_sub (xs : List ι) (f g : ι → R) :
    sumOn xs (fun i => f i - g i) = sumOn xs f - sumOn xs g := by
  induction xs with
  | nil => simp [sumOn]; grind
  | cons x xs ih => simp [sumOn, ih]; grind

theorem sumOn_add (xs : List ι) (f g : ι → R) :
    sumOn xs (fun i => f i + g i) = sumOn xs f + sumOn xs g := by
  induction xs with
  | nil => simp [sumOn]; grind
  | cons x xs ih => simp [sumOn, ih]; grind

theorem sumOn_congr (xs : List ι) (f g : ι → R)
    (same : ∀ i ∈ xs, f i = g i) : sumOn xs f = sumOn xs g := by
  induction xs with
  | nil => rfl
  | cons i xs ih =>
      have hFirst := same i (by simp)
      have hRest := ih (by intro j hj; exact same j (by simp [hj]))
      simp only [sumOn]; rw [hFirst, hRest]

theorem sumOn_scale (xs : List ι) (w : ι → R) (c : R) :
    sumOn xs (fun i => w i * c) = sumOn xs w * c := by
  induction xs with
  | nil => simp only [sumOn]; grind
  | cons i xs ih => simp only [sumOn, ih]; grind

theorem sumOn_flatten (blocks : List (List ι)) (f : ι → R) :
    sumOn blocks.flatten f = sumOn blocks (fun b => sumOn b f) := by
  induction blocks with
  | nil => rfl
  | cons b bs ih => simp [List.flatten_cons, sumOn_append, sumOn, ih]

/-- Scalar-source subtraction, fixed target and common information functional. -/
def xi (ei : List ι → R) (xs : List ι) : R :=
  ei xs - sumOn xs (fun i => ei [i])

theorem xi_singleton (ei : List ι → R) (i : ι) : xi ei [i] = 0 := by
  simp [xi, sumOn]; grind

theorem ei_budget (ei : List ι → R) (xs : List ι) :
    ei xs = sumOn xs (fun i => ei [i]) + xi ei xs := by
  simp only [xi]; grind

/-- S3.3: the binary residual is independent of singleton EI bookkeeping. -/
theorem binary_xi_residual (ei : List ι → R) (left right : List ι) :
    xi ei (left ++ right) - xi ei left - xi ei right =
      ei (left ++ right) - ei left - ei right := by
  simp only [xi, sumOn_append]; grind

def residual (v : List ι → R) (xs : List ι) (blocks : List (List ι)) : R :=
  v xs - sumOn blocks v

/-- S3.3: replace one block by its refinement, at any position in the enumeration.
The identity holds for arbitrary block lists, hence in particular for genuine
partitions and non-overlapping refinements. No partition fact is assumed away. -/
theorem hierarchical_refinement (v : List ι → R) (xs block : List ι)
    (before after refined : List (List ι)) :
    residual v xs (before ++ refined ++ after) =
      residual v xs (before ++ block :: after) + residual v block refined := by
  simp only [residual, sumOn_append, sumOn]; grind

theorem binary_hierarchy (v : List ι → R) (left right : List ι) :
    v (left ++ right) = v left + v right +
      residual v (left ++ right) [left, right] := by
  simp only [residual, sumOn]; grind

inductive BlockTree (ι : Type) where
  | leaf : List ι → BlockTree ι
  | fork : BlockTree ι → BlockTree ι → BlockTree ι

def members : BlockTree ι → List ι
  | .leaf xs => xs
  | .fork left right => members left ++ members right

def leafBudget (v : List ι → R) : BlockTree ι → R
  | .leaf xs => v xs
  | .fork left right => leafBudget v left + leafBudget v right

def internalBudget (v : List ι → R) : BlockTree ι → R
  | .leaf _ => 0
  | .fork left right =>
      (v (members left ++ members right) - v (members left) - v (members right)) +
      internalBudget v left + internalBudget v right

/-- Whole-tree telescoping retains nonzero grouped E/I leaf budgets. -/
theorem tree_budget (v : List ι → R) (tree : BlockTree ι) :
    v (members tree) = leafBudget v tree + internalBudget v tree := by
  induction tree with
  | leaf xs => simp only [members, leafBudget, internalBudget]; grind
  | fork left right ihLeft ihRight =>
      simp only [members, leafBudget, internalBudget]
      grind

/-- S1.3: increments along any ordering telescope, with the empty-set term explicit. -/
def marginalTotal (v : List ι → R) (acc : List ι) : List ι → R
  | [] => 0
  | i :: rest =>
      (v (acc ++ [i]) - v acc) + marginalTotal v (acc ++ [i]) rest

theorem marginal_telescope (v : List ι → R) (acc order : List ι) :
    marginalTotal v acc order = v (acc ++ order) - v acc := by
  induction order generalizing acc with
  | nil => simp only [marginalTotal, List.append_nil]; grind
  | cons i rest ih =>
      simp only [marginalTotal, ih, List.append_assoc, List.cons_append, List.nil_append]
      grind

theorem marginal_linearity (low high : List ι → R) (acc : List ι) (i : ι) :
    ((low (acc ++ [i]) + high (acc ++ [i])) - (low acc + high acc)) =
      (low (acc ++ [i]) - low acc) + (high (acc ++ [i]) - high acc) := by
  grind

/-- The sum of allocations is conserved for any normalized finite average
of complete orderings. This does not prove the factorial/subset-to-permutation
counting identity in the printed exact Shapley formula. -/
theorem averaged_marginal_efficiency (v : List ι → R) (fullValue : R)
    (orders : List (List ι)) (weight : List ι → R)
    (sameEnd : ∀ order ∈ orders, v order = fullValue)
    (normalized : sumOn orders weight = 1) (emptyZero : v [] = 0) :
    sumOn orders (fun order => weight order * marginalTotal v [] order) = fullValue := by
  have hEqual : sumOn orders (fun order => weight order * marginalTotal v [] order) =
      sumOn orders (fun order => weight order * fullValue) := by
    apply sumOn_congr
    intro order hOrder
    rw [marginal_telescope, List.nil_append, sameEnd order hOrder, emptyZero]
    grind
  rw [hEqual, sumOn_scale, normalized]
  grind

/-- S3.2: cancellation after factorization; the entropy facts are explicit inputs. -/
theorem entropy_cancellation (xs : List ι) (h hc : ι → R) (hJoint hcJoint : R)
    (independentEntropy : hJoint = sumOn xs h) :
    (hJoint - hcJoint) - sumOn xs (fun i => h i - hc i) = sumOn xs hc - hcJoint := by
  rw [sumOn_sub, independentEntropy]; grind

/-- S3.1: the co-information identity from seven finite entropy values. -/
theorem two_source_information_identity
    (hU hV hY hUV hUY hVY hUVY : R) :
    (hUV + hY - hUVY) - (hU + hY - hUY) - (hV + hY - hVY) =
      (hUY + hVY - hY - hUVY) - (hU + hV - hUV) := by
  grind

theorem independent_two_source_identity
    (hU hV hY hUV hUY hVY hUVY : R) (independentEntropy : hUV = hU + hV) :
    (hUV + hY - hUVY) - (hU + hY - hUY) - (hV + hY - hVY) =
      hUY + hVY - hY - hUVY := by
  grind

/-- S1.1: unnormalized determinism minus degeneracy, with a common reference. -/
theorem determinism_degeneracy (h0 hY hcY : R) :
    (h0 - hcY) - (h0 - hY) = hY - hcY := by grind

end Algebra

section FieldAlgebra
variable {R : Type} [Field R]

/-- S1.1: normalization requires a nonzero reference entropy. -/
theorem normalized_determinism (h0 hY hcY : R) (nonzero : h0 ≠ 0) :
    h0 * ((1 - hcY / h0) + hY / h0 - 1) = hY - hcY := by grind

end FieldAlgebra

section Ordered
variable {R : Type} [CommRing R] [LE R] [LT R]
    [Std.IsLinearOrder R] [Std.LawfulOrderLT R] [OrderedRing R]
variable {ι : Type}

theorem sumOn_nonnegative (xs : List ι) (f : ι → R)
    (nonnegative : ∀ i ∈ xs, 0 ≤ f i) : 0 ≤ sumOn xs f := by
  induction xs with
  | nil => simp [sumOn]
  | cons i xs ih =>
      have hFirst : 0 ≤ f i := nonnegative i (by simp)
      have hRest : 0 ≤ sumOn xs f := ih (by intro j hj; exact nonnegative j (by simp [hj]))
      simp only [sumOn]; grind

/-- S3.2 upper bound: only singleton-MI nonnegativity is needed. -/
theorem xi_upper_bound (ei : List ι → R) (xs : List ι)
    (singleNonnegative : ∀ i ∈ xs, 0 ≤ ei [i]) : xi ei xs ≤ ei xs := by
  have := sumOn_nonnegative xs (fun i => ei [i]) singleNonnegative
  simp only [xi]; grind

/-- Conditional result, not an independent formal proof of entropy subadditivity. -/
theorem xi_nonnegative_from_entropy (xs : List ι) (h hc : ι → R) (hJoint hcJoint : R)
    (independentEntropy : hJoint = sumOn xs h)
    (conditionalSubadditivity : hcJoint ≤ sumOn xs hc) :
    0 ≤ (hJoint - hcJoint) - sumOn xs (fun i => h i - hc i) := by
  rw [entropy_cancellation xs h hc hJoint hcJoint independentEntropy]; grind

/-- S4.1: maximizing the child sum and minimizing the residual are equivalent. -/
theorem score_residual_order (parent scoreA scoreB : R) :
    scoreA ≤ scoreB ↔ parent - scoreB ≤ parent - scoreA := by grind

/-- S3.1: N + PLM for a copied source pair + Id + source independence. -/
theorem redundancy_zero (redY redCopy sourceMI : R)
    (nonnegative : 0 ≤ redY) (copiedTargetPLM : redY ≤ redCopy)
    (identityProperty : redCopy = sourceMI) (independentSources : sourceMI = 0) :
    redY = 0 := by grind

omit [LE R] [LT R] [Std.IsLinearOrder R] [Std.LawfulOrderLT R] [OrderedRing R] in
theorem pid_correspondence (jointEI source1EI source2EI red unique1 unique2 syn : R)
    (redZero : red = 0)
    (unique1Def : unique1 = source1EI - red)
    (unique2Def : unique2 = source2EI - red)
    (pidDecomposition : jointEI = red + unique1 + unique2 + syn) :
    unique1 = source1EI ∧ unique2 = source2EI ∧ syn = jointEI - source1EI - source2EI := by
  grind

variable [DecidableLE R]

def minTwo (a b : R) : R := if a ≤ b then a else b
def witness (d a b : R) : R := minTwo d (minTwo a b)

theorem minTwo_bounds (a b : R) : minTwo a b ≤ a ∧ minTwo a b ≤ b := by
  unfold minTwo; split <;> grind

omit [CommRing R] [LT R] [Std.IsLinearOrder R] [Std.LawfulOrderLT R] [OrderedRing R] in
theorem minTwo_lower (a b x : R) (ha : x ≤ a) (hb : x ≤ b) : x ≤ minTwo a b := by
  unfold minTwo; split <;> grind

theorem minTwo_symmetry (a b : R) : minTwo a b = minTwo b a := by
  unfold minTwo; split <;> split <;> grind

theorem minTwo_monotone (a b a' b' : R) (ha : a ≤ a') (hb : b ≤ b') :
    minTwo a b ≤ minTwo a' b' := by
  unfold minTwo; split <;> split <;> grind

/-- S2: nonnegativity of the witness and all four induced PID components.
The inputs are ordinary MI nonnegativity, source-side data processing and
the co-information identity with a nonnegative conditional mutual information. -/
theorem witness_pid_nonnegative (d a b c cmi : R)
    (hd : 0 ≤ d) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (jointGeA : a ≤ c) (jointGeB : b ≤ c)
    (cmiNonnegative : 0 ≤ cmi) (coInformation : a + b - c = d - cmi) :
    0 ≤ witness d a b ∧ 0 ≤ a - witness d a b ∧
      0 ≤ b - witness d a b ∧ 0 ≤ c - a - b + witness d a b := by
  have hAB := minTwo_bounds a b
  have hDAB := minTwo_bounds d (minTwo a b)
  have hNonnegAB := minTwo_lower a b 0 ha hb
  have hNonneg := minTwo_lower d (minTwo a b) 0 hd hNonnegAB
  have hCoA : a + b - c ≤ a := by grind
  have hCoB : a + b - c ≤ b := by grind
  have hCoD : a + b - c ≤ d := by grind
  have hCoAB := minTwo_lower a b (a + b - c) hCoA hCoB
  have hCo := minTwo_lower d (minTwo a b) (a + b - c) hCoD hCoAB
  unfold witness; grind

theorem witness_symmetry (d a b : R) : witness d a b = witness d b a := by
  simp only [witness, minTwo_symmetry a b]

/-- S2: PLM of the witness follows from the two ordinary MI DPI inequalities. -/
theorem witness_target_processing (d aZ bZ aT bT : R)
    (dpi1 : aZ ≤ aT) (dpi2 : bZ ≤ bT) : witness d aZ bZ ≤ witness d aT bT := by
  exact minTwo_monotone _ _ _ _ (by grind) (minTwo_monotone _ _ _ _ dpi1 dpi2)

omit [CommRing R] [LT R] [Std.IsLinearOrder R] [Std.LawfulOrderLT R] [OrderedRing R] in
theorem witness_identity (d h1 h2 : R) (bound1 : d ≤ h1) (bound2 : d ≤ h2) :
    witness d h1 h2 = d := by
  have := minTwo_lower h1 h2 d bound1 bound2
  unfold witness minTwo; split <;> grind

omit [CommRing R] [LT R] [Std.LawfulOrderLT R] [OrderedRing R] in
theorem witness_source_inclusion (h1 a b : R) (entropyBound : a ≤ h1) (dpi : a ≤ b) :
    witness h1 a b = a := by
  unfold witness minTwo; split <;> split <;> grind

end Ordered

end PEID

#print axioms PEID.sumOn_append
#print axioms PEID.sumOn_sub
#print axioms PEID.sumOn_add
#print axioms PEID.sumOn_congr
#print axioms PEID.sumOn_scale
#print axioms PEID.sumOn_flatten
#print axioms PEID.xi_singleton
#print axioms PEID.ei_budget
#print axioms PEID.binary_xi_residual
#print axioms PEID.hierarchical_refinement
#print axioms PEID.binary_hierarchy
#print axioms PEID.tree_budget
#print axioms PEID.marginal_telescope
#print axioms PEID.marginal_linearity
#print axioms PEID.averaged_marginal_efficiency
#print axioms PEID.entropy_cancellation
#print axioms PEID.two_source_information_identity
#print axioms PEID.independent_two_source_identity
#print axioms PEID.determinism_degeneracy
#print axioms PEID.normalized_determinism
#print axioms PEID.sumOn_nonnegative
#print axioms PEID.xi_upper_bound
#print axioms PEID.xi_nonnegative_from_entropy
#print axioms PEID.score_residual_order
#print axioms PEID.redundancy_zero
#print axioms PEID.pid_correspondence
#print axioms PEID.minTwo_bounds
#print axioms PEID.minTwo_lower
#print axioms PEID.minTwo_symmetry
#print axioms PEID.minTwo_monotone
#print axioms PEID.witness_pid_nonnegative
#print axioms PEID.witness_symmetry
#print axioms PEID.witness_target_processing
#print axioms PEID.witness_identity
#print axioms PEID.witness_source_inclusion
