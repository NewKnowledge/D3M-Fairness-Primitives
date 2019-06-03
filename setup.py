from setuptools import setup

setup(name='FairnessPrimitives',
    version='1.0.0',
    description='Pre-processing, in-processing, and post-processing algorithms wrapped for the D3M infrastructure',
    packages=['FairnessPrimitives'],
    install_requires=["aif360 @ git+https://github.com/NewKnowledge/AIF360@7b6e9df57735fae0e97f6824e6abaed3e3f45e1a#egg=aif360",
                    "numba",
                    "BlackBoxAuditing",
                    "cvxpy"
                      ],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.data_conversion.FairnessPreProcessing = FairnessPrimitives:FairnessPreProcessing',
            'data_augmentation.data_conversion.FairnessInProcessing = FairnessPrimitives:in_processing',
            'data_augmentation.data_conversion.FairnessPostProcessing = FairnessPrimitives:post_processing',
       ],
    },
)
