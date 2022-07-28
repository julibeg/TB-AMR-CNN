import pathlib

from . import load_data
from . import preprocessing
from . import custom_metrics

# define paths and other constants
_PKG_ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent.parent
CRYPTIC_DATA_PATH = f"{_PKG_ROOT_PATH}/data/cryptic"
MAIN_DATA_PATH = f"{_PKG_ROOT_PATH}/data/main"
SAVED_MODELS_PATH = f"{_PKG_ROOT_PATH}/saved_models"

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
