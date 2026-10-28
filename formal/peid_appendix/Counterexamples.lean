import Lean

/-!
Exact counterexample to the unrestricted sign-invariance sentence in S4.2.
No random numbers, floating-point arithmetic, or added axioms are used.
-/
set_option autoImplicit false

namespace PEIDCounterexample

def direction (i : Nat) : Int := if i < 2 then -1 else 1

/-- Ascending values with the original index as the tie order. -/
def stableAscending (values : Nat → Int) (order : List Nat) : Prop :=
  order.Pairwise (fun i j => values i < values j ∨ (values i = values j ∧ i < j))

instance (values : Nat → Int) (order : List Nat) : Decidable (stableAscending values order) :=
  inferInstanceAs (Decidable (order.Pairwise _))

def positiveOrder : List Nat := [0, 1, 2, 3]
def negativeOrder : List Nat := [2, 3, 0, 1]

def bitMask (xs : List Nat) : Nat := xs.foldl (fun mask i => mask + 2 ^ i) 0

/-- Encode an unordered split by the smaller of its two complementary masks. -/
def unorderedPrefixesWith (fullMask : Nat) (order : List Nat) : List Nat :=
  (List.range (order.length - 1)).map fun k =>
    let mask := bitMask (order.take (k + 1))
    min mask (fullMask - mask)

def unorderedPrefixes (order : List Nat) : List Nat := unorderedPrefixesWith 15 order

theorem stable_orders :
    stableAscending direction positiveOrder ∧ stableAscending (fun i => -direction i) negativeOrder := by
  decide +kernel

theorem different_unordered_candidate_sets :
    1 ∈ unorderedPrefixes positiveOrder ∧ 1 ∉ unorderedPrefixes negativeOrder := by
  decide +kernel

/-- A symmetric, strictly positive off-diagonal affinity matrix with tied pairs. -/
def affinity (i j : Fin 4) : Rat :=
  if i = j then 0 else if (i.val < 2) = (j.val < 2) then 3 else 1

def dotRow (i : Fin 4) (v : Fin 4 → Rat) : Rat :=
  affinity i 0 * v 0 + affinity i 1 * v 1 + affinity i 2 * v 2 + affinity i 3 * v 3

def tiedDirection (i : Fin 4) : Rat := if i.val < 2 then -1 else 1

def laplacianRow (i : Fin 4) (v : Fin 4 → Rat) : Rat := v i - dotRow i v / 5

theorem affinity_degrees : ∀ i : Fin 4, dotRow i (fun _ => 1) = 5 := by
  decide +kernel

theorem tied_laplacian_eigenvector :
    ∀ i : Fin 4, laplacianRow i tiedDirection = (4 / 5 : Rat) * tiedDirection i := by
  decide +kernel

theorem tied_direction_nonconstant : tiedDirection 0 ≠ tiedDirection 2 := by
  decide +kernel

/-- First four exact Hadamard eigendirections, average-linkage ordering,
and original ordering of the connected 16-node example in check_spectral_sign.py. -/
def fullOrders : List (List Nat) :=
  [[1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14],
   [2,3,6,7,10,11,14,15,0,1,4,5,8,9,12,13],
   [1,2,5,6,9,10,13,14,0,3,4,7,8,11,12,15],
   [4,5,6,7,12,13,14,15,0,1,2,3,8,9,10,11],
   [0,8,12,4,6,14,10,2,3,11,15,7,5,13,9,1],
   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

def flippedFullOrders : List (List Nat) :=
  [0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15] :: fullOrders.tail

/-- Even after merging all six ordering pools, the unordered split
{0,2} | complement appears only after changing the first direction's sign. -/
theorem full_pool_sign_change :
    5 ∉ fullOrders.flatMap (unorderedPrefixesWith 65535) ∧
      5 ∈ flippedFullOrders.flatMap (unorderedPrefixesWith 65535) := by
  decide +kernel

end PEIDCounterexample

#print axioms PEIDCounterexample.stable_orders
#print axioms PEIDCounterexample.different_unordered_candidate_sets
#print axioms PEIDCounterexample.affinity_degrees
#print axioms PEIDCounterexample.tied_laplacian_eigenvector
#print axioms PEIDCounterexample.tied_direction_nonconstant
#print axioms PEIDCounterexample.full_pool_sign_change
