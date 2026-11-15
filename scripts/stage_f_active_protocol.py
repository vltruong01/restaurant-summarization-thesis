"""Dispatch current validated protocol while keeping pre-amendment modules historical."""
from stage_f_d031_protocol import active_d031
from stage_f_d030_protocol import relevance_eligible,unfiltered_fixed_seed_order


def current_module():
    from stage_f_d033_protocol import active_d033
    if active_d033():
        import stage_f_d033_protocol as module
        return module
    from stage_f_d032_protocol import active_d032
    if active_d032():
        import stage_f_d032_protocol as module
        return module
    if active_d031():import stage_f_d031_protocol as module
    else:import stage_f_d030_protocol as module
    return module


def require_validated_amendment():return current_module().require_validated_amendment()
def assert_scoring_binding(role,report):return current_module().assert_scoring_binding(role,report)
def assert_complete_hard_thresholds(frozen):return current_module().assert_complete_hard_thresholds(frozen)
def validate_compatibility():return current_module().validate_compatibility()
