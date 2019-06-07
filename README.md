# D3M-Fairness-Primitives

The base library (IBM's AIF360) containing all pre-processing, in-processing, and post-processing algorithms can be found here: https://github.com/NewKnowledge/AIF360

# FairnessPreProcessing
Primitive that applies one of three pre-processing algorithm to training data before fitting a learning algorithm. Algorithm options are *Disparate_Impact_Remover*, *Learning_Fair_Representations*, and *Reweighing*.

## Available Functions

#### produce (inputs)
Produce pre-processed D3M Dataframe according to some distance / fairness / representation / distribution
constraint defined by the algorithm. This pre-processing is only applied to training data and not to testing data.

# FairnessInProcessing
Primitive that applies an in-processing algorithm to training data while fitting a learning algorithm. Algorithm is *Adversarial_Debiasing*, which learns a classifier (tf nn based) that maximizes prediction accuracy, while simultaneously reducing an adversary’s ability to determine the protected attribute from the predictions.

## Available Functions

#### set_training_data (inputs, outputs)

Sets primitive's training data. The inputs are features and the outputs are labels

#### fit

Fits *Adversarial_Debiasing* classifier using training data from set_training_data and hyperparameters. There are no inputs or outputs.

#### produce (inputs)

Produce predictions using the fitted adversarial debiasing algorithm

# FairnessPostProcessing
Primitive that applies one of three post-processing algorithms after a classifier has been fit. This changes the predicted labels to achieve fairness according to some definition. The algorithm options 
(that is - fairness definition options) are *Calibrated_Equality_of_Odds*, *Equality_of_Odds*, and *Reject_Option_Classification*. The algorithm *Calibrated_Equality_of_Odds* can be modified by the 
hyperparameter *cost_constraint* (options = *weighted*, *fpr*, *fnr*). Additionally, the algorithm 
*Reject_Option_Classification* can be modified by the hyperparameters *metric_name* (options = 
*Statistical parity difference*, *Average odds difference*, *Equal opportunity difference*), 
*low_class_threshold* and *high_class_threshold*

## Available Functions

#### set_training_data (inputs, outputs)

Sets primitive's training data. The inputs are a dataframe containing a classifier's predicted labels. The outputs are a dataframe containing ground truth labels. 

#### fit

Fits post-processing primitive using training data from set_training_data and hyperparameters. There are no inputs or outputs.

#### produce
Produce predictions using the fitted post-processing algorithm


