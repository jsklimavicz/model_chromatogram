from samples import Sample, SampleCreator


sample_creator = SampleCreator()

samples = sample_creator.product_stability_samples(
    n_samples=10,
    timing_interval=10,
    compound_mapping=[
        {
            "intial_compound": "90094-11-4",
            "compounds": ["56843-54-0", "50-48-6", "117-89-5", "72-43-5", "79-11-8"],
            "k": [10, 10, 2, 2, 3],
            "initial_concentration": 10,
        },
        {
            "intial_compound": "56843-54-0",
            "compounds": ["56-54-2", "1617-90-9"],
            "k": [1, 3],
        },
        {
            "intial_compound": "50-48-6",
            "compounds": ["56-54-2", "93-76-5"],
            "k": [1, 1],
        },
        {
            "intial_compound": "117-89-5",
            "compounds": ["773-76-2", "72-43-5"],
            "k": [2, 3],
        },
        {
            "intial_compound": "79-11-8",
            "compounds": ["56-54-2", "93-76-5"],
            "k": [1, 1],
        },
    ],
)
