import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.metrics

# local import
import util


@tf.function
def masked_BCE_from_logits(y_true, y_pred_logits):
    """
    Computes the BCE loss from logits and tolerates NaNs in `y_true`.
    """
    non_nan_ids = tf.where(~tf.math.is_nan(y_true))
    y_true_non_nan = tf.gather_nd(y_true, non_nan_ids)
    y_pred_logits_non_nan = tf.gather_nd(y_pred_logits, non_nan_ids)
    return tf.nn.sigmoid_cross_entropy_with_logits(
        y_true_non_nan, y_pred_logits_non_nan
    )


@tf.function
def weighted_masked_BCE_from_logits(y_true, y_pred_logits, weights):
    """
    Applies weights after calculating the BCE loss from logits; tolerates NaNs in
    `y_true`.
    """
    non_nan_ids = tf.where(~tf.math.is_nan(y_true))
    y_true_non_nan = tf.gather_nd(y_true, non_nan_ids)
    y_pred_logits_non_nan = tf.gather_nd(y_pred_logits, non_nan_ids)
    weights_non_nan = tf.gather(weights, non_nan_ids[:, 1])
    return weights_non_nan * tf.nn.sigmoid_cross_entropy_with_logits(
        y_true_non_nan, y_pred_logits_non_nan
    )


@tf.function
def weighted_masked_mean_accuracy_from_logits(y_true, y_pred_logits, weights):
    """
    Gets the weighted accuracy between `y_true` and predicted logits in `y_pred_logits`
    while tolerating NaNs in `y_true`.
    """
    non_nan = ~tf.math.is_nan(y_true)
    matching_predictions = tf.equal(y_true, tf.round(tf.sigmoid(y_pred_logits)))
    weighted_acc_sum = tf.experimental.numpy.nansum(
        weights * tf.cast(matching_predictions, "float32")
    )
    weight_sum = tf.experimental.numpy.nansum(tf.cast(non_nan, "float32") * weights)
    return weighted_acc_sum / weight_sum


@tf.function
def nan_ACC(y_true, y_pred, from_logits=False):
    """
    Computes the accuracy between `y_true` and `y_pred`; tolerates NaNs in `y_true`.
    """
    nans = tf.reduce_sum(tf.cast(tf.math.is_nan(y_true), "float32"))
    y_pred = tf.round(tf.sigmoid(y_pred) if from_logits else y_pred)
    num_matching_predictions = tf.reduce_sum(
        tf.cast(tf.equal(y_true, y_pred), "float32")
    )
    return num_matching_predictions / (tf.cast(tf.size(y_true), "float32") - nans)


def nan_ACC_per_drug(y_true, y_pred, from_logits=False):
    """
    Expects `y_true` and `y_pred` to be 2D tensors of shape (num_samples, num_drugs).
    Returns a `pd.Series` with the accuracies for each drug while tolerating NaNs in
    `y_true`.
    """
    accs = pd.Series(index=util.DRUGS, data=np.nan)
    for i, drug in enumerate(util.DRUGS):
        accs[drug] = nan_ACC(
            y_true[:, i], y_pred[:, i], from_logits=from_logits
        ).numpy()
    return accs


@tf.function
def nan_AUC_from_logits(y_true, y_pred):
    """
    Computes the AUC between `y_true` and `y_pred`; tolerates NaNs in `y_true`.
    """
    non_nan_ids = tf.where(~tf.math.is_nan(y_true))
    y_true_non_nan = tf.gather(y_true, non_nan_ids)
    y_pred_non_nan = tf.sigmoid(tf.gather(y_pred, non_nan_ids))
    # there cannot be an AUC if y_true has only ones or zeros
    if tf.reduce_sum(y_true_non_nan) == 0.0 or tf.reduce_sum(y_true_non_nan) == tf.cast(
        len(y_true_non_nan), "float32"
    ):
        return np.nan
    # there was weird behaviour with tf.keras.metrics.AUC --> use the sklearn
    # implementation instead
    return tf.py_function(
        sklearn.metrics.roc_auc_score, (y_true_non_nan, y_pred_non_nan), "float32"
    )


def nan_AUC_per_drug_from_logits(y_true, y_pred, drugs):
    """
    Expects `y_true` and `y_pred` to be 2D tensors of shape (num_samples, num_drugs).
    Returns a `pd.Series` with the AUCs for each drug while tolerating NaNs in `y_true`.
    """
    aucs = pd.Series(index=drugs, data=np.nan)
    for i, drug in enumerate(drugs):
        aucs[drug] = nan_AUC_from_logits(y_true[:, i], y_pred[:, i]).numpy()
    return aucs


class nan_AUC_metric_per_drug_from_logits(tf.keras.metrics.Metric):
    """
    Metric that computes the AUC for each drug; needs to be initialized with a list of
    drugs.
    """

    def __init__(self, drugs, name="column_wise_NaN-aware_AUC", **kwargs):
        super().__init__(name=name, **kwargs)
        self.drugs = drugs
        # add a vector to keep track of the AUCs
        self.AUCs = self.add_weight(
            name="AUCs", shape=(len(self.drugs),), initializer="zeros"
        )
        # add a vector for the number of batches with valid AUCs per drug
        self.Ns = self.add_weight(
            name="Ns", shape=(len(self.drugs),), initializer="zeros"
        )

    def reset_state(self):
        # reset everything to zeros
        self.AUCs.assign(tf.zeros(shape=(len(self.drugs),)))
        self.Ns.assign(tf.zeros(shape=(len(self.drugs),)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Calculate AUCs for each drug for the current batch of samples and update the
        metric.
        """
        if sample_weight is not None:
            raise ValueError(f'Sample weights are not supported by "{self.name}" yet.')
        # get the AUC for each drug
        aucs = tf.map_fn(
            fn=lambda x: nan_AUC_from_logits(*x),
            elems=(tf.transpose(y_true), tf.transpose(y_pred)),
            fn_output_signature="float32",
        )
        # replace NaNs in the AUC vector with 0s before adding to the total
        nan_ids = tf.where(tf.math.is_nan(aucs))
        aucs = tf.tensor_scatter_nd_update(
            aucs, nan_ids, tf.zeros(tf.shape(nan_ids)[0])
        )
        self.AUCs.assign_add(aucs)
        # increase the counts for each drug that produced a valid AUC value
        incr_Ns = tf.tensor_scatter_nd_update(
            tf.ones_like(self.Ns), nan_ids, tf.zeros(tf.shape(nan_ids)[0])
        )
        self.Ns.assign_add(incr_Ns)

    def result(self):
        # return a dict with the mean AUC for each drug
        res_dict = {}
        for i, drug in enumerate(self.drugs):
            res_dict[drug] = self.AUCs[i] / self.Ns[i]
        return res_dict
