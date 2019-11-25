from setuptools import setup

setup(name='FairnessPrimitives',
    version='1.0.0',
    description='Pre-processing, in-processing, and post-processing algorithms wrapped for the D3M infrastructure',
    packages=['FairnessPrimitives'],
    install_requires=["aif360 @ git+https://github.com/NewKnowledge/AIF360.git@97cc0d45b33392d3d3fcadfd65c885b4b9fab6a2#egg=aif360",
                    'tensorflow-gpu==2.0.0',
                    'numba==0.46.0'
                      ],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.data_conversion.FairnessPreProcessing = FairnessPrimitives:FairnessPreProcessing',
            'data_augmentation.data_conversion.FairnessInProcessing = FairnessPrimitives:FairnessInProcessing',
            'data_augmentation.data_conversion.FairnessPostProcessing = FairnessPrimitives:FairnessPostProcessing',
       ],
    },
)
