"""Independent exact references and zero-coupling controls for the pilot."""
import unittest
import numpy as np

from scripts.dmf_joint_readout import CommonTargetGame, exact_shapley, audit_nonnegative
from scripts.dmf_subject_consistency import network_values, roi_shapley, transition_interval
from scripts.analyze_dmf_subject_consistency import loso, independent_loso


def linear_game(coefficient):
    d=len(coefficient)
    prior=np.eye(d)
    cross=prior@coefficient
    future=coefficient.T@cross+.3*np.eye(d)
    return CommonTargetGame(np.block([[prior,cross],[cross.T,future]]))


class SubjectConsistencyTests(unittest.TestCase):
    def test_population_zero_coupling_retains_local_but_no_cross_roi(self):
        # ROI blocks are (i,i+3). Only within-ROI mixing is allowed.
        coefficient=np.eye(6)
        for i in range(3):
            coefficient[i,i+3]=.6
            coefficient[i+3,i]=.2
        game=linear_game(coefficient)
        self.assertGreater(game.v(range(3)),0)
        self.assertAlmostEqual(game.u(range(3)),0,places=12)
        within,between,total,audit=network_values(game,[[0],[1],[2]])
        np.testing.assert_allclose(between,0,atol=1e-12)
        self.assertAlmostEqual(within.sum(),game.v(range(3)),places=12)

    def test_permutation_shapley_against_exact_small_game(self):
        rng=np.random.default_rng(8)
        game=linear_game(rng.normal(size=(6,6)))
        full,se,cross,audit=roi_shapley(game,pairs=4096)
        reference=exact_shapley(game,(0,1,2))['cross_shapley_nats']
        np.testing.assert_allclose(cross,reference,atol=5*se.max()+1e-12)
        self.assertAlmostEqual(full.sum(),game.v((0,1,2)),places=11)
        self.assertLess(audit['maximum_closure_error_nats'],1e-10)

    def test_network_game_with_roi_singletons_equals_exact_roi_game(self):
        rng=np.random.default_rng(11)
        game=linear_game(rng.normal(size=(6,6)))
        within,between,total,audit=network_values(game,[[0],[1],[2]])
        reference=exact_shapley(game,(0,1,2))['cross_shapley_nats']
        np.testing.assert_allclose(between,reference,atol=1e-12)

    def test_nonnegative_violation_is_explicit(self):
        with self.assertRaisesRegex(ArithmeticError,'threshold=.*affected=1'):
            audit_nonnegative([0,-2e-8])
        self.assertEqual(audit_nonnegative([-1e-9])['tolerance_negative_count'],1)

    def test_boundary_transition_is_not_force_aligned(self):
        self.assertFalse(transition_interval([0,1,2,3],[0,1,2,8])['located'])
        self.assertTrue(transition_interval([0,1,2,3],[0,1,8,9])['located'])

    def test_cross_seed_loso_removes_identical_shared_noise(self):
        rng=np.random.default_rng(37)
        shared=rng.normal(size=(3,100))
        values=np.tile(shared,(8,1,1))
        paired,_=loso(values.mean(1))
        independent,_=independent_loso(values)
        np.testing.assert_allclose(paired,1,atol=1e-12)
        self.assertLess(abs(independent.mean()),.2)


if __name__=='__main__':
    unittest.main()
