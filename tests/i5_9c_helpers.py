from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    build_i5_9a_remaining_depth_provider_robustness_admission,
)
from core.reconstruction.i5_remaining_depth_provider_robustness_replay import (
    build_i5_9b1_replay,
)
from core.reconstruction.i5_relative_depth_provider_robustness_replay import (
    build_i5_9b2_replay,
)
from tests.i5_9a_helpers import build_args as i59a_build_args
from tests.i5_9b1_helpers import build_replay_args as i59b1_build_replay_args
from tests.i5_9b2_helpers import build_replay_args as i59b2_build_replay_args

I59A_SHA = "50ae4a51d57390e6b20b16aa2fa83c65b110a5ba29803853af279ffa3c4a7cff"
I59B1_SHA = "e1e66f6bebcc6b06cc6b48861da093234f3e7e9ca4c593b81569f243756585fa"
I59B2_SHA = "b" * 64
I59C_RECEIPT_SHA = "c" * 64


def clone(value):
    return deepcopy(value)


def binding(path: str, sha256: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha256)


def admission() -> dict:
    return build_i5_9a_remaining_depth_provider_robustness_admission(**i59a_build_args())


def b1_replay() -> dict:
    result = build_i5_9b1_replay(**i59b1_build_replay_args())
    for providers in result['records'].values():
        for record in providers.values():
            record['fusion_capacity_drops'] = 0
            record['accepted_views'] = record['selected_views']
    return result


def b2_replay() -> dict:
    args = i59b2_build_replay_args(overrides={('castle_dynamic_proxy', 'marigold'): 'degraded'})
    # Bind the synthetic B1 predecessor exactly the same way the real B2 artifact does.
    args['i5_9b1_binding'] = binding('/evidence/i5_9b1.json', I59B1_SHA)
    args['runtime_preflight']['evidence_bindings']['i5_9b1_replay']['sha256'] = I59B1_SHA
    result = build_i5_9b2_replay(**args)
    dynamic = result['records']['castle_dynamic_proxy']['marigold']
    dynamic['accepted_views'] = 18
    dynamic['fusion_capacity_drops'] = 0
    dynamic['native_metrics'].update({
        'alignment_accepted_fraction': 18 / 19,
        'alignment_median_relative_residual': 0.008962760233577727,
        'canonical_point_count': 681510,
        'peak_vram_mib': 2778.0,
        'wall_time_s': 118.20037890295498,
    })
    dynamic['execution_result'] = {
        'status': 'degraded',
        'degradation_reasons': ['alignment_rejected_views:1'],
        'failure_reason': '',
    }
    for sid, providers in result['records'].items():
        for provider, record in providers.items():
            record['fusion_capacity_drops'] = 0
            if not (sid == 'castle_dynamic_proxy' and provider == 'marigold'):
                record['accepted_views'] = record['selected_views']
                record['native_metrics']['alignment_accepted_fraction'] = 1.0
                record['execution_result'] = {
                    'status': 'completed',
                    'degradation_reasons': [],
                    'failure_reason': '',
                }
    return result


def receipt() -> dict:
    return {
        'schema_version': 1,
        'phase': 'I5.9C_QUALIFICATION_RECEIPT',
        'status': 'pass',
        'pass': True,
        'scope': {
            'focused_i5_9c_tests': True,
            'i5_9a_admission_regression': True,
            'i5_9b1_replay_regression': True,
            'i5_9b2_replay_regression': True,
            'historical_g4_preservation_regression': True,
            'mixed_terminal_quality_regression': True,
            'authority_guard_regression': True,
            'roadmap_guard_regression': True,
            'import_probe': True,
            'runner_help_probe': True,
            'provider_inference': False,
            'sfm_rerun': False,
            'mvs_rerun': False,
            'provider_comparison_started': False,
            'thresholds_modified': False,
            'fusion_capacity_increased': False,
            'production_default_modified': False,
            'result_driven_retuning': False,
            'retry_until_pass': False,
            'mp_sfm_started': False,
            'auto_policy_implemented': False,
            'hybrid_authorized': False,
        },
        'counts': {
            'focused_tests': 1,
            'predecessor_unit_tests': 1,
            'predecessor_integration_tests': 1,
            'predecessor_stress_tests': 1,
            'predecessor_total_tests': 3,
            'combined_exercised_tests': 4,
        },
        'return_codes': {
            'focused': 0,
            'predecessor_unit': 0,
            'predecessor_integration': 0,
            'predecessor_stress': 0,
            'import_probe': 0,
            'runner_help_probe': 0,
        },
    }


def build_args() -> dict:
    return {
        'i5_9a_admission': admission(),
        'i5_9a_binding': binding('/evidence/i5_9a.json', I59A_SHA),
        'i5_9b1_replay': b1_replay(),
        'i5_9b1_binding': binding('/evidence/i5_9b1.json', I59B1_SHA),
        'i5_9b2_replay': b2_replay(),
        'i5_9b2_binding': binding('/evidence/i5_9b2.json', I59B2_SHA),
        'qualification_receipt': receipt(),
        'qualification_receipt_binding': binding('/evidence/i5_9c_receipt.json', I59C_RECEIPT_SHA),
    }
