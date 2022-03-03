# Metrics function
from collections import OrderedDict
from aif360.metrics import ClassificationMetric
import pandas as pd 


## function takend out of the Aif 360 code to get x and y out of the dataset.
def extract_df_from_ds(dataset):
    """Extract data frames from Transformer Data set
    Args:
         :param dataset: aif360 dataset
    Returns:
         :return X, X_prime, y: pandas dataframes of attributes, sensitive attributes, labels
    """
    X = pd.DataFrame(dataset.convert_to_dataframe()[0])
    # remove labels
    X = X.drop(columns=dataset.label_names)
    # get sensitive attributes
    X_prime = X[dataset.protected_attribute_names]
    y = tuple(dataset.labels[:, 0])
    return X, X_prime, y


def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics




def rewrite_german_dataset(dataset):
    
    df_german_data_set, attributes = dataset.convert_to_dataframe()
    df_german_data_set['credit'] = df_german_data_set['credit'] - 1 # rewrite labels to 0 and 1
    
    fixed_german_data_set = StructuredDataset(df_german_data_set, label_names = attributes['label_names'],\
                                        protected_attribute_names = attributes['protected_attribute_names'],\
                                      unprivileged_protected_attributes = attributes['unprivileged_protected_attributes'],\
                                      privileged_protected_attributes = attributes['privileged_protected_attributes'] )
    return fixed_german_data_set
