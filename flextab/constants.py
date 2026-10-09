from enum import StrEnum

# The average norm needs to be computed offline on a representative set of strings
# (e.g. all strings in the CARTE validation set, or whatever)
# If using sentence_transformers, they should already normalize results to norm 1.0
text_embedding_model_properties = {
    'sentence-transformers/all-MiniLM-L6-v2': {'dimension': 384, 'pooling': 'mean', 'average_norm': 5.0}
}

# labels get indices from 0 to MAX_CLASSIFICATION_CLASSES - 1 (inclusive)
# index MAX_CLASSIFICATION_CLASSES is reserved for everything else
# for target column in masking those indices are increased by 1 to indicate that index 0 is a mask
MAX_CLASSIFICATION_CLASSES = 62

# Bin indices explanation:
# index 0 is centered at quantile_0 = the min value of the train distribution
# index 1 is centered at quantile_{1/2n} where n = NUM_REGRESSION_BINS_EXCLUDING_TAILS
# ...
# index n is centered at quantile_{(2n-1)/2n}
# index n+1 is centered at quantile_1 = the max value of the train distribution
# index n+2 is reserved for nan values for non target columns
# Value are encoded as linear combinations of the two closest bin centers
# (for query rows with values outside the train distribution range the closest bin center is used)
NUM_REGRESSION_BINS_EXCLUDING_TAILS = 50
NUM_VALID_BINS = NUM_REGRESSION_BINS_EXCLUDING_TAILS + 2
NAN_BIN = NUM_VALID_BINS
TOTAL_NUM_REGRESSION_BINS = NUM_REGRESSION_BINS_EXCLUDING_TAILS + 3

# max text length to embed
MAX_TEXT_LENGTH = 512


class TaskType(StrEnum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    MATCHING = 'matching'
    ANOMALY_DETECTION = 'anomaly_detection'
    CLUSTERING = 'clustering'
    MULTI_TABLE = 'multi_table'
