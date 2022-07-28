import tensorflow as tf


def get_model(
    filter_length=25,
    num_filters=64,
    num_conv_layers=2,
    num_dense_layers=2,
    dense_neurons=256,
    conv_dropout_rate=0.2,
    dense_dropout_rate=0.5,
    bias_before_batchnorm=False,
    return_logits=False,
):
    # input layer
    input = tf.keras.Input(shape=(None, 4), name="input")
    # feature extraction
    x = tf.keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        data_format="channels_last",
        use_bias=bias_before_batchnorm,
        name="extract_features",
    )(input)
    x = tf.keras.layers.BatchNormalization(name="extract_features_BN")(x)
    x = tf.keras.layers.Activation("relu", name="extract_features_RELU")(x)
    # convolution layers
    for i in range(1, num_conv_layers + 1):
        x = tf.keras.layers.Dropout(conv_dropout_rate, name=f"conv{i}_dropout")(x)
        x = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=3,
            data_format="channels_last",
            use_bias=bias_before_batchnorm,
            name=f"conv{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"conv{i}_BN")(x)
        x = tf.keras.layers.Activation("relu", name=f"conv{i}_RELU")(x)
        x = tf.keras.layers.MaxPool1D(3, name=f"conv{i}_mp")(x)
    # feature combination
    x = tf.keras.layers.GlobalMaxPooling1D(name="combine_features")(x)
    # dense layers
    for i in range(1, num_dense_layers + 1):
        x = tf.keras.layers.Dropout(dense_dropout_rate, name=f"d{i}_dropout")(x)
        x = tf.keras.layers.Dense(
            units=dense_neurons, use_bias=bias_before_batchnorm, name=f"d{i}_dense"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"d{i}_BN")(x)
        x = tf.keras.layers.Activation("relu", name=f"d{i}_RELU")(x)
    # ouptut layer --> split into Dense and Activation so that only the sigmoid uses
    # the more precise float32 dtype.
    if return_logits:
        predictions = tf.keras.layers.Dense(
            units=13, dtype="float32", name="dense_predict"
        )(x)
    else:
        x = tf.keras.layers.Dense(units=13, name="dense_predict")(x)
        predictions = tf.keras.layers.Activation(
            "sigmoid", dtype="float32", name="predictions"
        )(x)
    return tf.keras.Model(inputs=input, outputs=predictions)
