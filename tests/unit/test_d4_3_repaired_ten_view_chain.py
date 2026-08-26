from __future__ import annotations
import pytest
from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_d4_3_repaired_ten_view_chain import (
    _candidate_workspace_orders,
    _select_repaired_candidate,
    _validate_source_evidence,
)


def _d4():
    return {'phase':'D4','pass':True,'views':[{'workspace_order':i} for i in range(6,16)]}

def _d41():
    return {'phase':'D4.1','pass':False,'status':'degraded'}

def _d42(classification='pair_specific_incompatibility'):
    return {
        'phase':'D4.2','diagnostic_complete':True,
        'diagnosis':{'classification':classification,'preferred_single_view_exclusion_workspace_order':None},
        'weak_adjacent_pair':{'workspace_orders':[7,8]},
        'bridge_pairs':[{'role':'left','status':'pass'},{'role':'right','status':'pass'}],
    }


def _candidate(pass_value: bool, consistency: float, median: float):
    pair={'a_to_b':{'consistent_ratio_of_comparable':consistency,'median_absolute_relative_depth_residual':median},
          'b_to_a':{'consistent_ratio_of_comparable':consistency,'median_absolute_relative_depth_residual':median}}
    return {'pass':pass_value,'adjacent_pairs':[pair for _ in range(9)]}


def test_validate_source_evidence_accepts_pair_specific_7_8_case():
    assert _validate_source_evidence(_d4(),_d41(),_d42()) == (7,8)


def test_validate_source_evidence_rejects_non_pair_specific_or_bad_bridge():
    with pytest.raises(ReconstructionValidationError):
        _validate_source_evidence(_d4(),_d41(),_d42('span_instability'))
    payload=_d42(); payload['bridge_pairs'][0]['status']='degraded'
    with pytest.raises(ReconstructionValidationError):
        _validate_source_evidence(_d4(),_d41(),payload)


def test_candidate_workspace_orders_match_minimal_repairs():
    result=_candidate_workspace_orders(range(6,16),predecessor_order=5,weak_left_order=7,weak_right_order=8)
    assert result['A'] == (5,6,8,9,10,11,12,13,14,15)
    assert result['B'] == (5,6,7,9,10,11,12,13,14,15)


def test_candidate_workspace_orders_reject_invalid_source():
    with pytest.raises(ReconstructionValidationError):
        _candidate_workspace_orders(range(6,15),predecessor_order=5,weak_left_order=7,weak_right_order=8)


def test_select_repaired_candidate_handles_no_or_single_pass():
    selected, reason=_select_repaired_candidate({'A':_candidate(False,.9,.03),'B':_candidate(False,.9,.03)})
    assert selected is None and reason=='no_candidate_qualified'
    selected, reason=_select_repaired_candidate({'A':_candidate(True,.8,.04),'B':_candidate(False,.9,.03)})
    assert selected=='A' and reason=='only_passing_candidate'


def test_select_repaired_candidate_prefers_stronger_worst_case_consistency():
    selected, reason=_select_repaired_candidate({'A':_candidate(True,.65,.05),'B':_candidate(True,.82,.06)})
    assert selected=='B'
    assert reason=='best_dense_consistency_among_passing_candidates'


def test_select_repaired_candidate_uses_worst_median_as_tie_breaker():
    selected,_=_select_repaired_candidate({'A':_candidate(True,.80,.08),'B':_candidate(True,.80,.05)})
    assert selected=='B'
