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
from aif360.algorithms import preprocessing
from aif360.algorithms.preprocessing.optim_preproc_helpers import opt_tools

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'Disparate_Impact_Remover', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['Disparate_Impact_Remover', 'Learning_Fair_Representations', 'Optimized_Preprocessing', 'Reweighing'],
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
        description='label value which is considered favorable (i.e. positive)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    pass

class FairnessPreProcessing(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies one of four pre-processing algorithm to training data before fitting step of pipeline. Algorithm
        options are 'Disparate_Impact_Remover', 'Learning_Fair_Representations', 'Optimized_Preprocessing', and 'Reweighing'.

        Fit method applies primitive pre-processing. Produce method leaves dataset as is. 

    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "20736e8c-4f8c-484d-b128-33aa6fb20549",
        'version': __version__,
        'name': "Pre-processing Fairness Techniques",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['fairness, bias, debias, data preprocessing, data augmentation'],
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
        'python_path': 'd3m.primitives.data_preprocessing.data_conversion.FairnessPreProcessing',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self.attributes = None
        self.targets = None
        self.values = None
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
        self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(self.targets):
            self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(self.targets):
            self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        label_names = [list(inputs)[t] for t in self.targets]
        
        # calculate protected attributes and privel=
        protected_attributes = [list(inputs)[c] for c in self.hyperparams['protected_attribute_cols']]

        # save index and metadata
        idx = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        idx = [list(inputs)[i] for i in idx]
        index = inputs[idx]
        
        # mark attributes that are not priveleged data
        attributes = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        priveleged_data = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrivilegedData')
        self.attributes = list(set(attributes) - set(priveleged_data))
        attribute_names = [list(inputs)[a] for a in self.attributes]
        
        # drop index from training data
        inputs = inputs.drop(columns=idx)

        # transfrom dataframe to IBM 360 compliant dataset
            # 1. assume datacleaning primitive has been applied so there are no NAs
            # 2. assume PandasOneHotEncoderPrimitive has also been applied to categorical columns
        unfavorable_label = 0. if self.hyperparams['favorable_label'] == 1. else 1.
        ibm_dataset = datasets.BinaryLabelDataset(df = inputs,
                                                label_names = label_names,
                                                protected_attribute_names = protected_attributes,
                                                favorable_label=self.hyperparams['favorable_label'],
                                                unfavorable_label=unfavorable_label)

        # apply pre-processing algorithm
        if self.hyperparams['algorithm'] == 'Disparate_Impact_Remover':
            transformed_dataset = preprocessing.DisparateImpactRemover().fit_transform(ibm_dataset)
        elif self.hyperparams['algorithm'] == 'Learning_Fair_Representations':
            transformed_dataset = preprocessing.LFR(unprivileged_groups = [{protected_attributes[0]: ibm_dataset.unprivileged_protected_attributes}],
                                                                privileged_groups = [{protected_attributes[0]: ibm_dataset.privileged_protected_attributes}]).fit_transform(ibm_dataset)
        elif self.hyperparams['algorithm'] == 'Optimized_Preprocessing':
            transformed_dataset = preprocessing.OptimPreproc(unprivileged_groups = [{protected_attributes[0]: ibm_dataset.unprivileged_protected_attributes}],
                                                                privileged_groups = [{protected_attributes[0]: ibm_dataset.privileged_protected_attributes}],
                                                                optimizer = opt_tools.OptTools,
                                                                optim_options = {}).fit_transform(ibm_dataset)
        else: 
            privileged_groups = [{p_attr: p_attr_val} for (p_attr, p_attr_val) in zip(protected_attributes, ibm_dataset.privileged_protected_attributes)]
            unprivileged_groups = [{p_attr: p_attr_val} for (p_attr, p_attr_val) in zip(protected_attributes, ibm_dataset.unprivileged_protected_attributes)]
            transformed_dataset = preprocessing.Reweighing(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups).fit_transform(ibm_dataset)
            # TODO: incorporate instance weights fro transformed_dataset.instance_weights into classifier

        # transform IBM dataset back to D3M dataset
        df = transformed_dataset.convert_to_dataframe()[0].drop(columns = label_names)
        df = d3m_DataFrame(pandas.concat([index.reset_index(drop=True), inputs[label_names].reset_index(drop = True), df.reset_index(drop=True)], axis = 1))
        df.metadata = inputs.metadata
        self.values = df

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
        
        hp_class = random_forest.RandomForestClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'] 
        hp=hp_class.defaults().replace({'use_inputs_columns': self.attributes, 'use_outputs_columns': self.targets})
        self.clf = random_forest.RandomForestClassifierPrimitive(hyperparams=hp)
        self.clf.set_training_data(inputs = self.values, outputs = self.values)
        return self.clf.fit()

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
        return self.clf.produce(inputs = inputs)

