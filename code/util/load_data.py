import pandas as pd
import glob
import numpy as np
from Bio import SeqIO

# local import
import util


def read_sequence_data(dir_path):
    """
    Reads a sequence dataset from a directory with FASTA files (one file per locus --
    with multiple sequences for the multiple samples). Splits the descpription line at
    '/' and expects the last field to be the sample ID.
    """
    seqs_df = pd.DataFrame()
    for file in glob.glob(f"{dir_path}/fasta_files/*.fasta"):
        locus = file.split("/")[-1].split(".")[0].split("_")[0]
        seqs_df[locus] = pd.Series(
            {
                record.id.split("/")[-1]
                .replace(".cut", ""): str(record.seq)
                .replace("-", "")
                .upper()
                for record in SeqIO.parse(file, "fasta")
            }
        )
    return seqs_df


def get_cryptic_dataset():
    """
    Get sequences and resistance phenotypes for the CRyPTIC dataset
    """
    res = pd.read_csv(
        f"{util.CRYPTIC_DATA_PATH}/phen/cryptic_dataset.csv", index_col="ROLLINGDB_ID"
    )
    res = (
        res[[col for col in res.columns if col in util.DRUGS]]
        .replace({"R": 1, "S": 0, "I": np.nan})
        .astype(float)
    ).dropna(how="all")
    seqs = read_sequence_data(util.CRYPTIC_DATA_PATH)
    return seqs.loc[res.index], res


def get_main_dataset():
    """
    Get sequences and resistance phenotypes for the "main" dataset
    """
    res = pd.read_csv(
        f"{util.MAIN_DATA_PATH}/phen/master_table_resistance.csv",
        low_memory=False,
        index_col=0,
        usecols=lambda x: x != "index",
    )
    res = res[util.DRUGS].replace({"R": 1, "S": 0}).astype(float).dropna(how="all")
    seqs = read_sequence_data(util.MAIN_DATA_PATH)
    # get the ids of samples for which we have sequences and phenotypes
    ids = seqs.index.intersection(res.index)
    return seqs.loc[ids], res.loc[ids]
