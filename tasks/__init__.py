"""
BEREAN Tasks Module
===================

Status: NEW
Source: berean_v2_refactor

Custom MAMMAL tasks for antibody-antigen prediction.

These tasks follow the official MAMMAL task pattern from biomed-multi-alignment
and can be used with the official mammal.main_finetune training infrastructure.

Tasks:
    - AntibodyHABindingTask: Binary classification for Ab-HA binding
    - HAIPredictionTask: Regression for HAI titer prediction
"""

from .antibody_ha_binding import AntibodyHABindingTask
from .hai_prediction import HAIPredictionTask

__all__ = [
    "AntibodyHABindingTask",
    "HAIPredictionTask",
]
