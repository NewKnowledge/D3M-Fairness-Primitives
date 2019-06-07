from setuptools import setup

setup(name='FairnessPrimitives',
    version='1.0.0',
    description='Pre-processing, in-processing, and post-processing algorithms wrapped for the D3M infrastructure',
    packages=['FairnessPrimitives'],
    install_requires=["aif360 @ git+https://github.com/NewKnowledge/AIF360@0b64efa87cefba89ac5a29f80c7eb885a80a1422#egg=aif360",
                    'tensorflow-gpu<=1.12.2',
                      ],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.data_conversion.FairnessPreProcessing = FairnessPrimitives:FairnessPreProcessing',
            'data_augmentation.data_conversion.FairnessInProcessing = FairnessPrimitives:FairnessInProcessing',
            'data_augmentation.data_conversion.FairnessPostProcessing = FairnessPrimitives:FairnessPostProcessing',
       ],
    },
)
