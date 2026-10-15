import numpy as np

from scripts.spt_glv_a_comparison import splits,hac,quality,gates


def test_bipartitions_are_complete_without_duplicates():
    result=list(splits(511))
    assert len(result)==255
    assert len({frozenset((l,r)) for l,r in result})==255
    assert all(l&r==0 and l|r==511 for l,r in result)


def test_average_hac_recovers_separated_modules_and_handles_signed_weights():
    matrix=np.full((9,9),-.0001)
    for block in [(0,1,2,3,4),(5,6,7,8)]:
        matrix[np.ix_(block,block)]=.1
    for block in [(0,1,2),(3,4),(5,6),(7,8)]:
        matrix[np.ix_(block,block)]=1.
    np.fill_diagonal(matrix,0.)
    result=hac(matrix)
    assert result['clade_f1']==1. and result['correct_root']
    assert abs(result['strict_clade_f1']-12/13)<1e-12
    assert hac(matrix*100)['clades']==result['clades']


def test_precision_failures_are_explicit():
    assert gates(dict(ess_minimum=1,maximum_normalized_weight=.8,max_projected_step=.1))==[
        'ESS<10','weight>=.25','proposal_step>=1e-3']
