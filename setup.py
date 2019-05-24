from setuptools import setup

setup(name='D3MFairnessPrimitives',
    version='1.0.0',
    description='Pre-processing, in-processing, and post-processing algorithms wrapped for the D3M infrastructure',
    packages=['D3MFairnessPrimitives'],
    install_requires=["aif360",
                      ],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.data_conversion.FairnessPreProcessing = D3MFairnessPrimitives:pre_processing',
            'data_augmentation.data_conversion.FairnessInProcessing = D3MFairnessPrimitives:in_processing',
            'data_augmentation.data_conversion.FairnessPostProcessing = D3MFairnessPrimitives:post_processing',
       ],
    },
)
