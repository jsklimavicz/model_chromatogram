from samples import Sample, SampleCreator
import numpy as np


sample_creator = SampleCreator()

compound_mapping = [
    {
        "initial_compound": "90094-11-4",
        "compounds": [
            ["56843-54-0", "50-48-6"],
            ["117-89-5", "72-43-5"],
            "79-11-8",
        ],
        "k": [0.00100, 0.002, 0.0032],
        "initial_concentration": 10,
    },
    {
        "initial_compound": "56843-54-0",
        "compounds": [["56-54-2", "1617-90-9"]],
        "k": [0.0072],
    },
    {
        "initial_compound": "50-48-6",
        "compounds": ["56-54-2", "93-76-5", "79-11-8"],
        "k": [0.01, 0.04, 0.002],
    },
    {
        "initial_compound": "117-89-5",
        "compounds": ["773-76-2", "72-43-5"],
        "k": [0.0006, 0.0003],
    },
    {
        "initial_compound": "79-11-8",
        "compounds": ["56-54-2"],
        "k": [0.01],
    },
]


sample_creator.product_stability_samples(
    time_points=np.array(
        [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 105, 120, 150, 180]
    ),
    compound_mapping=compound_mapping,
)
