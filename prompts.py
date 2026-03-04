# prompts.py
from __future__ import annotations

# Variant 2 : Cars (Zero-Shot)
# IMPORTANT: les clés (noms de classes) doivent correspondre aux noms des dossiers du dataset.

PROMPTS_BY_CLASS: dict[str, list[str]] = {
    "oil_change": [
        "how to change engine oil",
        "oil change tutorial for a car",
        "changing the oil filter on a car",
        "draining and refilling car engine oil",
    ],
    "tire_change": [
        "how to change a tire",
        "replacing a wheel on a car",
        "tire change tutorial",
        "installing a spare tire",
    ],
    "car_wash_detailing": [
        "car wash and detailing",
        "cleaning a car interior",
        "polishing and waxing a car",
        "washing a car step by step",
    ],
    "car_review": [
        "car review video",
        "test driving and reviewing a car",
        "reviewing a new car model",
        "car comparison review",
    ],
}

# Optionnel : templates de prompts (si tu veux générer automatiquement)
PROMPT_TEMPLATES: list[str] = [
    "a video about {label}",
    "a tutorial about {label}",
]