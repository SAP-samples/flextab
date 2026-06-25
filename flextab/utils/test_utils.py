import re


def _sample_negatives(pos_pairs, left_ids, right_ids, neg_ratio, rng):
    """Sample non-matching pairs, avoiding the known positive set."""
    n_neg = len(pos_pairs) * neg_ratio
    pos_set = set(map(tuple, pos_pairs))
    neg_left_ids: list = []
    neg_right_ids: list = []
    while len(neg_left_ids) < n_neg:
        l_ids = rng.choice(left_ids, size=n_neg - len(neg_left_ids))
        r_ids = rng.choice(right_ids, size=n_neg - len(neg_left_ids))
        for l, r in zip(l_ids, r_ids):
            if (l, r) not in pos_set:
                neg_left_ids.append(l)
                neg_right_ids.append(r)
                if len(neg_left_ids) >= n_neg:
                    break
    return neg_left_ids, neg_right_ids


def _load_febrl4_pairs(neg_ratio: int = 5, seed: int = 42):
    """Build aligned (left, right, y) pairs from the FEBRL4 record-linkage dataset.

    Positive pairs come from the dataset's `true_links` MultiIndex; negatives are
    sampled uniformly while excluding the true links.
    """
    import numpy as np
    import pandas as pd
    from recordlinkage.datasets import load_febrl4

    dfA, dfB, true_links = load_febrl4(return_links=True)
    dfA = dfA.drop(columns=['soc_sec_id'], errors='ignore')
    dfB = dfB.drop(columns=['soc_sec_id'], errors='ignore')

    pos_left_ids = true_links.get_level_values(0)
    pos_right_ids = true_links.get_level_values(1)

    rng = np.random.RandomState(seed)
    neg_left_ids, neg_right_ids = _sample_negatives(
        list(zip(pos_left_ids, pos_right_ids)), dfA.index.values, dfB.index.values, neg_ratio, rng
    )

    all_left = pd.concat(
        [dfA.loc[pos_left_ids].reset_index(drop=True), dfA.loc[neg_left_ids].reset_index(drop=True)]
    ).reset_index(drop=True)
    all_right = pd.concat(
        [dfB.loc[pos_right_ids].reset_index(drop=True), dfB.loc[neg_right_ids].reset_index(drop=True)]
    ).reset_index(drop=True)
    n_pos = len(pos_left_ids)
    y = pd.Series([1] * n_pos + [0] * len(neg_left_ids), name='match')

    perm = rng.permutation(len(y))
    return (
        all_left.iloc[perm].reset_index(drop=True),
        all_right.iloc[perm].reset_index(drop=True),
        y.iloc[perm].reset_index(drop=True),
    )


def load_febrl4():
    """Load train/test split of the FEBRL4 matching dataset."""
    from sklearn.model_selection import train_test_split

    left, right, y = _load_febrl4_pairs()
    left_train, left_test, right_train, right_test, y_train, y_test = train_test_split(
        left, right, y, test_size=0.2, random_state=42, stratify=y
    )
    return (
        left_train.reset_index(drop=True),
        left_test.reset_index(drop=True),
        right_train.reset_index(drop=True),
        right_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def _clean_str(s: str) -> str:
    r"""Clean a single Fodors/Zagats string value.

    The raw CSVs wrap values in backticks/quotes and contain stray backslashes,
    e.g. ` gallagher \'s ` -> `gallagher's`.
    """
    s = s.replace('\\', '')
    s = s.strip(" `'")
    s = re.sub(r'\s{2,}', ' ', s)
    return s


def _clean_strings(df):
    """Apply _clean_str to every object column of a DataFrame."""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].where(df[col].isna(), df[col].map(_clean_str))
    return df


def _load_fodors_zagat_tables_and_pairs():
    """Download DeepMatcher Fodors-Zagats tables and all pair index splits."""
    import pandas as pd

    BASE = 'https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/exp_data/'
    DROP = ['id', 'class']

    tableA = _clean_strings(pd.read_csv(BASE + 'tableA.csv')).set_index('id').drop(columns=DROP, errors='ignore')
    tableB = _clean_strings(pd.read_csv(BASE + 'tableB.csv')).set_index('id').drop(columns=DROP, errors='ignore')

    train = pd.read_csv(BASE + 'train.csv')
    valid = pd.read_csv(BASE + 'valid.csv')
    test = pd.read_csv(BASE + 'test.csv')

    return tableA, tableB, train, valid, test


def _make_pairs(tableA, tableB, idx_df):
    left = tableA.loc[idx_df['ltable_id']].reset_index(drop=True)
    right = tableB.loc[idx_df['rtable_id']].reset_index(drop=True)
    y = idx_df['label'].rename('match').reset_index(drop=True)
    return left, right, y


def load_fodors_zagat():
    """Load the DeepMatcher Fodors-Zagats fixed train/test split.

    Merges train+valid as training context, uses the held-out test set for evaluation.
    946 total pairs with curated hard negatives from the DeepMatcher benchmark.
    """
    import pandas as pd

    tableA, tableB, train, valid, test = _load_fodors_zagat_tables_and_pairs()
    train_all = pd.concat([train, valid], axis=0).reset_index(drop=True)
    left_train, right_train, y_train = _make_pairs(tableA, tableB, train_all)
    left_test, right_test, y_test = _make_pairs(tableA, tableB, test)
    return left_train, left_test, right_train, right_test, y_train, y_test


def load_fodors_zagat_all():
    """Load all 946 DeepMatcher Fodors-Zagats pairs as a single pool for k-fold CV.

    Returns left, right, y with all train+valid+test pairs merged, matching the
    evaluation protocol used in the original DeepMatcher paper (5-fold CV on the full pool).
    """
    import pandas as pd

    tableA, tableB, train, valid, test = _load_fodors_zagat_tables_and_pairs()
    all_pairs = pd.concat([train, valid, test], axis=0).reset_index(drop=True)
    left, right, y = _make_pairs(tableA, tableB, all_pairs)
    return left, right, y


def _load_and_split_breast_cancer(invert_y=False):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    if invert_y:
        # For anomaly detection, the sklearn dataset uses a different convention 0 = anomaly.
        y = 1 - y
    return train_test_split(X, y, test_size=0.5, random_state=42)


def hungarian_accuracy(y_true, pred):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    K = max(len(np.unique(pred)), int(y_true.max()) + 1)
    conf = confusion_matrix(y_true, pred, labels=range(K))
    col_ind = np.argsort(linear_sum_assignment(-conf)[1])
    matched = col_ind[pred]
    return float((matched == y_true).mean())
