import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split

import util
import model

# define constants
EPOCHS = 100
# set random seed for improved reproducibility (perfect reproducibility with GPUs is
# impossible because of randomness intrinsic in some low-level GPU optimisations). We'll
# define a variable so that it can also be used for `pd.DataFrame.sample()` below)
RANDOM_SEED = 123
tf.random.set_seed(RANDOM_SEED)
# decide whether the loss function (BCE) should take the number of observations for each
# drug into account (i.e. give a higher weight to drugs with fewer resistance
# phenotypes). In our experience, there was not much difference and models with the
# unweighted loss actually converged faster and generalised better.
WEIGHTED_LOSS = False

# use mixed precision to improve efficiency
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# load training data
seqs_df, res_all = util.load_data.get_main_dataset()
N_samples = seqs_df.shape[0]
DRUGS = util.DRUGS
assert set(DRUGS) == set(res_all.columns)
N_drugs = len(DRUGS)

# load the CRyPTIC samples as test data
seqs_cryptic, res_cryptic = util.load_data.get_cryptic_dataset()
# make sure the loci are in the same order as in the training data
seqs_cryptic = seqs_cryptic[seqs_df.columns]

# split the training data into train and validation sets. To achieve some stratification
# by resistance phenotype, sort the drugs inversely by the number of observations and
# designate each sample with the least common drug it has a phenotype for.
drugs_sorted = res_all.isna().sum(axis=0).sort_values(ascending=False).index
least_common_drug_per_sample = pd.Series(pd.NA, index=res_all.index)
for idx, row in res_all.iterrows():
    for drug in drugs_sorted:
        if not pd.isna(row[drug]):
            least_common_drug_per_sample[idx] = drug
            break

# use this now to create the stratified train/val split
train_idx, val_idx = train_test_split(
    least_common_drug_per_sample.index,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=least_common_drug_per_sample,
)


# we will need tf datasets and one-hot encoded sequences
def get_seq_and_phen(seqs, res):
    """
    Generator to create a `tf.data.Dataset`. Yields sequences and their corresponding
    phenotypes. Uses the index of `res` to select sequences from `seqs`.
    """
    for idx, phen in res.iterrows():
        yield seqs.loc[idx].str.cat(), phen


# create the train tf dataset
train_ds = tf.data.Dataset.from_generator(
    lambda: get_seq_and_phen(seqs_df, res_all.loc[train_idx]),
    output_shapes=((), (13,)),
    output_types=(tf.string, "float32"),
)
# one-hot-encode the sequences
train_ds = train_ds.map(
    lambda seq, phen: (util.preprocessing.seq_to_one_hot(seq), phen),
    num_parallel_calls=16,
)
# now the same for validation
val_ds = tf.data.Dataset.from_generator(
    lambda: get_seq_and_phen(seqs_df, res_all.loc[val_idx]),
    output_shapes=((), (13,)),
    output_types=(tf.string, "float32"),
)
val_ds = val_ds.map(
    lambda seq, phen: (util.preprocessing.seq_to_one_hot(seq), phen),
    num_parallel_calls=16,
)
# similarly, prepare the cryptic dataset for testing later
ds_cryptic = tf.data.Dataset.from_generator(
    lambda: get_seq_and_phen(seqs_cryptic, res_cryptic),
    output_shapes=((), (8,)),
    output_types=(tf.string, "float32"),
)
ds_cryptic = ds_cryptic.map(
    lambda seq, phen: (util.preprocessing.seq_to_one_hot(seq), phen),
    num_parallel_calls=16,
)

# define the model
m = model.get_model(
    filter_length=25,
    num_filters=64,
    num_conv_layers=2,
    num_dense_layers=2,
    dense_neurons=256,
    conv_dropout_rate=0,
    dense_dropout_rate=0.2,
    bias_before_batchnorm=True,
    return_logits=True,
)

# define the loss function (BCE; weighted or unweighted)
if WEIGHTED_LOSS:
    # get weights inversely proportional to the number of observations per drug
    weights = N_samples / (N_samples - res_all.isna().sum(0))
    weights /= weights.min()
    weights = tf.constant(weights, dtype="float32")

    @tf.function
    def loss(y_true, y_pred):
        return util.custom_metrics.weighted_masked_BCE_from_logits(
            y_true, y_pred, weights
        )

else:
    loss = util.custom_metrics.masked_BCE_from_logits

# compile the model
opt = tf.keras.optimizers.Adam()
m.compile(
    optimizer=opt,
    loss=loss,
    metrics=[util.custom_metrics.nan_AUC_metric_per_drug_from_logits(util.DRUGS)],
)

# create a directory to save the model checkpoints etc.
model_version = "0.1"
model_dir = f"{util.SAVED_MODELS_PATH}/v{model_version}_FINAL_TEST"
pathlib.Path(model_dir).mkdir(exist_ok=True, parents=True)

# fit the model
batch_size = 256
prefetch_num = 2
# start with a relatively high learning rate which will be lowered whenever a plateau is
# reached
lr = 0.01
m.optimizer.learning_rate.assign(lr)
h1 = m.fit(
    train_ds.padded_batch(batch_size).cache().prefetch(prefetch_num),
    epochs=EPOCHS,
    validation_data=val_ds.padded_batch(batch_size).cache().prefetch(prefetch_num),
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=5, min_lr=0.0001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{model_dir}/model." "epoch-{epoch:03d}.savedmodel"
        ),
    ],
)

# save the training history
history = pd.DataFrame(h1.history)
pd.DataFrame(history).to_csv(f"{model_dir}/training-history.csv")

# plot the training history
fig, ax = plt.subplots()
ax.plot(history["loss"])
ax.plot(history["val_loss"])
ax_2 = ax.twinx()
ax_2.plot(history["lr"], "k--", lw=1)
ax_2.set_yscale("log")
ax.set_ylim(ax.get_ylim()[0], history["loss"][0])
ax.grid(axis="x")
fig.tight_layout()
fig.savefig(f"{model_dir}/training-history.png")
