# Higher Std for Carbonate values

class Config:
    NAME = "6"
    INPUT_XLSX = "../data/terrherbmarin.xlsx"
    SHEET_NAME = 0
    NROWS = 88
    SEED = 0
    FEATURES = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]
    SIZE = 100
    IBI_SIZE = 15
    IBI_STD_FACTOR = 1.0

Manual_Adjustments = {
    "herbivorous": {
        "δ13C coll": {"mean_shift": 0.0, "sigma_factor": 1},
        "δ15N coll": {"mean_shift": -1.0, "sigma_factor": 1},
        "δ13C carb": {"mean_shift": -2.0, "sigma_factor": 1},
        "δ18O carb": {"mean_shift": -2.0, "sigma_factor": 1},
    },
    "piscivorous": {
        "δ13C coll": {"mean_shift": 0.0, "sigma_factor": 1},
        "δ15N coll": {"mean_shift": 1.0, "sigma_factor": 1},
        "δ13C carb": {"mean_shift": 1.0, "sigma_factor": 1},
        "δ18O carb": {"mean_shift": 2.0, "sigma_factor": 1},
    },
    "ibi": {
        "δ13C coll": {"mean_shift": 0.0, "sigma_factor": 1.0},
        "δ15N coll": {"mean_shift": 0.0, "sigma_factor": 1.0},
        "δ13C carb": {"mean_shift": 0.4, "sigma_factor": 2.0},
        "δ18O carb": {"mean_shift": 0.4, "sigma_factor": 2.0},
    }
}