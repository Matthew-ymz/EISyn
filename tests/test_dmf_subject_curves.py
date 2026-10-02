"""Scientific controls for descriptive DMF curve agreement and native WMS."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.analyze_dmf_subject_curves import agreement, repeatability, standard_curve
from scripts.run_dmf_subject_curve_baselines import native_wms


class CurveControls(unittest.TestCase):
    def test_shape_agreement_ignores_positive_scale_and_offset(self):
        shape=np.array([0.,2.,3.,2.,1.,.5,-1.])
        curves=np.array([[shape*(i+1)*(j+1)+100*i-7*j for j in range(3)] for i in range(8)])
        result=agreement(curves)
        self.assertAlmostEqual(result['pearson_mean'],1.)
        self.assertAlmostEqual(result['spearman_mean'],1.)
        self.assertAlmostEqual(repeatability(curves)['pearson_mean'],1.)

    def test_shared_seed_curve_cannot_leak_into_training(self):
        # Three mutually orthogonal centered shapes; people share only the seed.
        q,_=np.linalg.qr(np.column_stack((np.ones(7),np.eye(7)[:,:3])))
        curves=np.repeat(q[:,1:4].T[None,:,:],8,axis=0)
        # Same-seed cross-person correlation is 1; independent-seed LOSO is 0.
        self.assertAlmostEqual(agreement(curves)['pearson_mean'],0.,places=12)

    def test_flat_and_nonfinite_curves_are_not_reported_as_consistent(self):
        for curve in [np.ones(7),np.array([1.,2.,np.nan,4.,5.,6.,7.])]:
            with self.assertRaises(ValueError): standard_curve(curve)

    def test_native_correlated_source_wms_is_signed(self):
        rng=np.random.default_rng(8927)
        source_cov=np.array([[1.,.8],[.8,1.]])
        x=rng.multivariate_normal([0.,0.],source_cov,size=100000)
        y=x+np.sqrt(.2)*rng.normal(size=x.shape)
        conditional=np.linalg.inv(np.linalg.inv(source_cov)+np.eye(2)/.2)
        joint=.5*np.log(np.linalg.det(source_cov)/np.linalg.det(conditional))
        singles=.5*np.log(np.diag(source_cov)/np.diag(conditional)).sum()
        expected=(joint-singles)/np.log(2)
        with threadpool_limits(limits=1):
            result,audit=native_wms(x,y)
        self.assertLess(result['raw_phi'],0.)
        self.assertAlmostEqual(result['raw_phi'],expected,delta=.015)
        self.assertEqual(audit['eigenvalue_floor_count'],0)
        self.assertFalse(result['factorized_source_covariance'])


if __name__=='__main__': unittest.main()
