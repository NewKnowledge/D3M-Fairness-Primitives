import sys
import os.path
import pandas
from typing import List

from d3m.primitive_interfaces.base import CallResult, PrimitiveBase

from d3m import container, utils, exceptions
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import random_forest

from aif360 import datasets, algorithms
from aif360.algorithms import inprocessing
import tensorflow as tf

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'Adversarial_Debiasing', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['Adversarial_Debiasing', 'ART_Classifier', 'Prejudice_Remover'],
        description = 'type of fairness pre-processing algorithm to use')
    protected_attribute_cols = hyperparams.List(
        elements=hyperparams.Hyperparameter[int](-1),
        default=[],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to use as protected attributes.",
    )
    favorable_label = hyperparams.Bounded[float](
        lower=0.,
        upper=1., 
        default=1.,
        description='label value which is considered favorable (i.e. positive) in the binary label case',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    pass

class FairnessInProcessing(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies one of three in-processing algorithm to training data while fitting a learning algorithm. Algorithm
        options are 'Adversarial_Debiasing' and 'Prejudice_Remover'.
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "f9822847-d19f-40f9-8e23-3fdcd5dcb847",
        'version': __version__,
        'name': "In-processing Fairness Techniques",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['fairness, bias, debias, data inprocessing, data augmentation'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/D3M-Fairness-Primitives",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [
             {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/D3M-Fairness-Primitives.git@{git_commit}#egg=FairnessPrimitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_augmentation.data_conversion.FairnessInProcessing',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_AUGMENTATION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self.label_names = None
        self.protected_attributes = None
        self.idx = None
        self.attribute_names = None
        self.unfavorable_label = None
        self.train_dataset = None
        self.clf = None

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params:Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data by applying pre-processing algorithm

        Parameters
        ----------
        inputs : features
        outputs : labels
        '''
                                                
        # only select attributes from training data
        targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        self.label_names = [list(inputs)[t] for t in targets]
        
        # calculate protected attributes 
        self.protected_attributes = [list(inputs)[c] for c in self.hyperparams['protected_attribute_cols']]

        # save index and metadata
        idx = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        self.idx = [list(inputs)[i] for i in idx]
        
        # mark attributes that are not priveleged data
        attributes = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        priveleged_data = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrivilegedData')
        attributes = list(set(attributes) - set(priveleged_data))
        self.attribute_names = [list(inputs)[a] for a in attributes]
        
        # transfrom dataframe to IBM 360 compliant dataset
            # 1. assume datacleaning primitive has been applied so there are no NAs
            # 2. assume categorical columns have been converted to unique numeric values
            # 3. assume the label column is numeric 
        self.unfavorable_label = 0. if self.hyperparams['favorable_label'] == 1. else 1.
        self.train_dataset = datasets.BinaryLabelDataset(df = inputs[self.attribute_names + self.label_names],
                                                label_names = self.label_names,
                                                protected_attribute_names = self.protected_attributes,
                                                favorable_label=self.hyperparams['favorable_label'],
                                                unfavorable_label=self.unfavorable_label)

        # apply pre-processing algorithm
        if self.hyperparams['algorithm'] == 'Adversarial_Debiasing':
            self.clf = inprocessing.AdversarialDebiasing(unprivileged_groups = [{self.protected_attributes[0]: self.train_dataset.unprivileged_protected_attributes}],
                                                                    privileged_groups = [{self.protected_attributes[0]: self.train_dataset.privileged_protected_attributes}],
                                                                    scope_name = 'adversarial_debiasing', sess = tf.Session())
        else:
            self.clf = inprocessing.Prejudice_Remover()

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit primitive using sklearn random forest on pre-processed training data

        Parameters
        ----------
        inputs : D3M dataframe

        Returns
        ----------
        Outputs : D3M dataframe unchanged
        """
        
        self.clf = self.clf.fit(self.train_dataset)
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce predictions using sklearn random forest 

        Parameters
        ----------
        inputs : D3M dataframe

        Returns
        ----------
        Outputs : predictions from sklearn random forest which was fit on pre-processed training data
            
        """
        # transfrom test dataframe to IBM 360 compliant dataset
        inputs[self.label_names] = self.train_dataset.convert_to_dataframe()[0][self.label_names].values[:inputs.shape[0]].astype(int)
        test_dataset = datasets.BinaryLabelDataset(df = inputs[self.attribute_names + self.label_names],
                                                label_names = self.label_names,
                                                protected_attribute_names = self.protected_attributes,
                                                favorable_label=self.hyperparams['favorable_label'],
                                                unfavorable_label=self.unfavorable_label)

        transformed_dataset = self.clf.predict(test_dataset)
        
        # transform IBM dataset back to D3M dataset
        df = transformed_dataset.convert_to_dataframe()[0][self.label_names].astype(int)
        df = d3m_DataFrame(pandas.concat([inputs[self.idx].reset_index(drop=True), df.reset_index(drop=True)], axis = 1))
        df.metadata = df.metadata.update((metadata_base.ALL_ELEMENTS, 0), inputs.metadata.query_column(0))
        df.metadata = df.metadata.update((metadata_base.ALL_ELEMENTS, 1), inputs.metadata.query_column(1))
        print(df.head(), file = sys.__stdout__)
        return CallResult(df)

