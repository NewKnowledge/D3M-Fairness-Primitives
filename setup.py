from setuptools import setup

setup(name='FairnessPrimitives',
    version='1.0.0',
    description='Pre-processing, in-processing, and post-processing algorithms wrapped for the D3M infrastructure',
    packages=['FairnessPrimitives'],
    install_requires=["aif360 @ git+https://github.com/NewKnowledge/AIF360@94e97ad2d73202e646922d706d53b3ed0bef4561#egg=aif360",
                    'tensorflow-gpu<=1.12.2',
                    "adversarial-robustness-toolbox @ git+https://github.com/NewKnowledge/adversarial-robustness-toolbox@14e11015cc710a851f839de5bacfb6b6f84ade81#egg=art"
                      ],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.data_conversion.FairnessPreProcessing = FairnessPrimitives:FairnessPreProcessing',
            'data_augmentation.data_conversion.FairnessInProcessing = FairnessPrimitives:FairnessInProcessing',
            'data_augmentation.data_conversion.FairnessPostProcessing = FairnessPrimitives:post_processing',
       ],
    },
)
