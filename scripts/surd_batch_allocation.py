"""Batched SURD allocation with the existing order/mask tie convention.

Implements SURD SI (S13)-(S16) at each target realization, then sums atoms.
Within each order, only increments above the preceding order's maximum survive.
For valid specific KL tables those order maxima are nondecreasing (marginal KL
data processing). This is algebraically the existing scalar allocator, not an
MI-before-allocation shortcut. Zero increments below the lower-order maximum
are the SURD allocation rule; estimated PEID Syn is never projected here.
"""
from __future__ import annotations

import numpy as np


def allocate_batch(specific, tolerance=1e-8):
    specific = np.asarray(specific,dtype=float)
    if specific.ndim!=2 or len(specific)==0:
        raise ValueError('Expected a nonempty target-by-coalition specific-KL table')
    width = specific.shape[1]
    n = width.bit_length()-1
    if width!=1 << n or not np.isfinite(specific).all():
        raise ValueError('SURD requires every source subset and finite specific KL')
    minimum = float(specific.min())
    violation = int(np.count_nonzero(specific < -tolerance))
    if violation:
        raise ArithmeticError(f'Specific KL: minimum={minimum} nats; threshold={-tolerance}; '
                              f'affected_count={violation}')
    masks = np.arange(width)
    sizes = np.array([int(m).bit_count() for m in masks])
    singleton_masks = 1 << np.arange(n)
    ranking = np.argsort(specific[:,singleton_masks],axis=1,kind='stable')
    values = np.take_along_axis(specific[:,singleton_masks],ranking,axis=1)
    increments = np.diff(values,axis=1,prepend=np.zeros((len(specific),1)))
    ranked_masks = singleton_masks[ranking]
    removed = np.concatenate((np.zeros((len(specific),1),dtype=int),
                              np.bitwise_xor.accumulate(ranked_masks,axis=1)[:,:-1]),axis=1)
    remaining = (width-1) ^ removed
    redundancy = np.bincount(remaining.ravel(),weights=increments.ravel(),minlength=width)
    synergy = np.zeros(width)
    state_total = values[:,-1].copy()
    lower_max = values[:,-1,None]
    minimum_order_margin = float('inf')
    for order in range(2,n+1):
        order_masks = masks[sizes==order]
        raw = specific[:,order_masks]
        ranking = np.argsort(raw,axis=1,kind='stable')
        ordered = np.take_along_axis(raw,ranking,axis=1)
        previous = np.concatenate((np.zeros((len(specific),1)),ordered[:,:-1]),axis=1)
        margin = float(np.min(ordered[:,-1,None]-lower_max))
        minimum_order_margin = min(minimum_order_margin,margin)
        if margin < -tolerance:
            raise ArithmeticError(f'Specific-KL order maxima decrease: minimum={margin} nats')
        # The three cases in the published specific-synergy definition (S15).
        increments = np.where(previous>=lower_max,ordered-previous,
                              np.where(ordered>lower_max,ordered-lower_max,0.))
        synergy += np.bincount(order_masks[ranking].ravel(),weights=increments.ravel(),minlength=width)
        state_total += increments.sum(axis=1)
        lower_max = ordered[:,-1,None]
    closure = float(np.max(abs(state_total-specific[:,-1])))
    if closure>tolerance:
        raise ArithmeticError(f'SURD statewise closure failed: {closure} nats')
    return {'redundancy_sum':redundancy,'synergy_sum':synergy,
            'mi_sum':specific.sum(axis=0),'samples':len(specific),
            'audit':{'minimum_specific_nats':minimum,
                     'specific_tolerance_negative_count':int(np.count_nonzero(specific<0)),
                     'specific_significant_violation_count':violation,
                     'maximum_statewise_closure_error_nats':closure,
                     'minimum_order_maximum_margin_nats':minimum_order_margin}}
