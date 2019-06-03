from setuptools import setup

setup(name='FairnessPrimitives',
    version='1.0.0',
    description='Pre-processing, in-processing, and post-processing algorithms wrapped for the D3M infrastructure',
    packages=['FairnessPrimitives'],
    install_requires=["aif360 @ git+https://github.com/NewKnowledge/AIF360@77c15436d6711f1d5fd44039a14e8d2825b6c63f#egg=aif360",
                      ],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.data_conversion.FairnessPreProcessing = FairnessPrimitives:FairnessPreProcessing',
            'data_augmentation.data_conversion.FairnessInProcessing = FairnessPrimitives:in_processing',
            'data_augmentation.data_conversion.FairnessPostProcessing = FairnessPrimitives:post_processing',
       ],
    },
)
