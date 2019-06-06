from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: Denormalize primitive
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: dataset_to_dataframe
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_hyperparameter(name='dataframe_resource', argument_type= ArgumentType.VALUE, data='learningData')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: DISTIL/NK data cleaning
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.data_cleaning.Datacleaning'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Step 3: column_parser
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.DataFrameCommon'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_output('produce')
pipeline_description.add_step(step_3)

# Step 4: imputer
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
step_4.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE,data='replace')
step_4.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE,data=True)
pipeline_description.add_step(step_4)

# Step 5: Fairness pre-processing Distil
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_preprocessing.data_conversion.FairnessPreProcessing'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
step_5.add_hyperparameter(name='algorithm', argument_type=ArgumentType.VALUE,data='Learning_Fair_Representations')
step_5.add_hyperparameter(name='protected_attribute_cols', argument_type=ArgumentType.VALUE,data=[3])
step_5.add_hyperparameter(name='favorable_label', argument_type=ArgumentType.VALUE,data=0.)
pipeline_description.add_step(step_5)

# Step 6: Random Forest classifier
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.random_forest.DataFrameCommon'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_output('produce')
step_6.add_hyperparameter(name='use_inputs_columns', argument_type=ArgumentType.VALUE,data=[2, 3, 4, 5, 6, 7])
step_6.add_hyperparameter(name='use_outputs_columns', argument_type=ArgumentType.VALUE,data=[1])
pipeline_description.add_step(step_6)

# Step 7: construct output
step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')

# Output to JSON
with open('pipeline.json', 'w') as outfile:
    outfile.write(pipeline_description.to_json())
