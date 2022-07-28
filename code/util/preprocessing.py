import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype="int64"),
        tf.TensorSpec(shape=(), dtype="int32"),
    ]
)
def get_tensor_with_ones_at_ids(ids, length):
    """
    Returns a tensor of length `length` with ones at the indices in `ids` and zeros
    everywhere else.
    """
    return tf.scatter_nd(
        ids,
        tf.ones(len(ids)),
        [tf.cast(length, tf.int64)],
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype="string"),
    ]
)
def seq_to_one_hot(seq):
    """
    Returns a tensor of shape `[len(seq), 4]` holding the one hot-encoded representation
    of the DNA sequence in `seq`.
    """
    seq_chars = tf.strings.bytes_split(seq)
    seq_one_hot = tf.concat(
        [
            tf.expand_dims(
                get_tensor_with_ones_at_ids(
                    tf.where(seq_chars == base),
                    len(seq_chars),
                ),
                axis=1,
            )
            for base in [b"A", b"C", b"G", b"T"]
        ],
        axis=1,
    )
    return seq_one_hot
