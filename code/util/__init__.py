import pathlib

from . import load_data
from . import preprocessing
from . import custom_metrics

# define paths and other constants
CRYPTIC_DATA_PATH = (
    f"{pathlib.Path(__file__).resolve().parent.parent.parent}/data/cryptic"
)
MAIN_DATA_PATH = f"{pathlib.Path(__file__).resolve().parent.parent.parent}/data/main"
SAVED_MODELS_PATH = (
    f"{pathlib.Path(__file__).resolve().parent.parent.parent}/saved_models"
)

DRUGS = [
    "AMIKACIN",
    "CAPREOMYCIN",
    "CIPROFLOXACIN",
    "ETHAMBUTOL",
    "ETHIONAMIDE",
    "ISONIAZID",
    "KANAMYCIN",
    "LEVOFLOXACIN",
    "MOXIFLOXACIN",
    "OFLOXACIN",
    "PYRAZINAMIDE",
    "RIFAMPICIN",
    "STREPTOMYCIN",
]
