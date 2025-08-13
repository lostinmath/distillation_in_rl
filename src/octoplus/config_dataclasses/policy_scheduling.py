from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class PolicySchedulingConfig:
    
    text_instruction: str
    """text insrtruction for External Agent"""
    
    scheduling_strategy: str = "internal_only"
    # octo_epsilon_decreasing octo_than_internal internal_octo_interchangeably 
    # # octo_reward_based octo_epsilon
    # Different parameters for differnt scheduling policies
    # has a lot of combinations
    
    # for octo_than_internal  
    # acts octo ... octo(step_to_switch times) internal ...internal within 1 run
    # after iteration_to_switch iterations acts only internal
    
    # ========= octo_than_internal ========== #
    iteration_to_switch: int = 50 
    step_to_switch: int = 15
    # for epsilon_octo
    epsilon: Optional[float] = 0.1
    policy_trust_length: Optional[int] = 5
    
    # decrease_type_epsilon:Optional[str] = "linear"
    decrease_until_global_step: Optional[int] = 500_000
    
    # ========== Reward based ============ #
    policy_trust_threshold: Optional[float] = 0.6
    internal_policy_trust_length: Optional[int] = 5
    internal_policy_warmup_length: Optional[int] = 5
    """at the begining we trust the internal policy more for this amount of steps?"""
    
    def __to_dict__(self) -> Dict[str, Any]:
        return self.__dict__