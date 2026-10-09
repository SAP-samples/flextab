import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from flextab.utils.test_utils import _load_and_split_breast_cancer, hungarian_accuracy, load_febrl4


def test_classification():
    from flextab import FlexTabClassifier

    print('CLASSIFICATION')

    X_train, X_test, y_train, y_test = _load_and_split_breast_cancer()

    # Initialize a classifier, 8k context and 8-fold bagging gives best performance, reduce if running out of memory
    clf = FlexTabClassifier(max_context_size=8192, bagging=8)

    clf.fit(X_train, y_train)

    # Predict probabilities
    prediction_probabilities = clf.predict_proba(X_test)
    # Predict labels
    predictions = clf.predict(X_test)
    print(f'Accuracy {accuracy_score(y_test, predictions):.2%}')
    print(f'AUROC {roc_auc_score(y_test, prediction_probabilities[:, 1]):.2%}\n')


def test_regression():
    from flextab import FlexTabRegressor

    print('REGRESSION')

    X, y = load_diabetes(return_X_y=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Initialize the regressor, 8k context and 8-fold bagging gives best performance, reduce if running out of memory
    regressor = FlexTabRegressor(max_context_size=8192, bagging=8)

    regressor.fit(X_train, y_train)

    # Predict on the test set
    predictions = regressor.predict(X_test)

    r2 = r2_score(y_test, predictions)
    print(f'R² Score: {r2:.2%}\n')


def test_matching():
    from flextab import FlexTabMatcher

    print('\nMATCHING')

    left_train, left_test, right_train, right_test, y_train, y_test = load_febrl4()

    matcher = FlexTabMatcher(max_context_size=1024, bagging=1)
    matcher.fit(left_train, right_train, y_train)

    probas = matcher.predict_proba(left_test, right_test)
    preds = matcher.classes_[probas.argmax(axis=-1)]
    print(f'Accuracy {accuracy_score(y_test, preds):.2%}')
    print(f'F1 Score {f1_score(y_test, preds):.2%}')
    print(f'AUROC {roc_auc_score(y_test, probas[:, 1]):.2%}')


def test_unsupervised_anomaly_detection():
    from flextab import FlexTabADUnsupervised

    print('ANOMALY DETECTION')

    X_train, X_test, _, y_test = _load_and_split_breast_cancer(True)

    unsup = FlexTabADUnsupervised()

    unsup.fit(X_train)
    scores = unsup.predict_score(X_test)
    print(f'Unsupervised AUROC = {roc_auc_score(y_test, scores):.2%}')


def test_oneclass_anomaly_detection():
    from flextab import FlexTabADOneClass

    X_train, X_test, y_train, y_test = _load_and_split_breast_cancer(True)
    train_mask = y_train == 0
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    oneclass = FlexTabADOneClass()
    oneclass.fit(X_train)
    scores = oneclass.predict_score(X_test)
    print(f'One-class AUROC = {roc_auc_score(y_test, scores):.2%}')


def test_semisupervised_anomaly_detection():
    from flextab import FlexTabADSemisupervised, FlexTabClassifier

    LA_RATIOS = [0.01, 0.05, 0.10, 0.50, 1.00]

    flextab = FlexTabClassifier()
    semisup = FlexTabADSemisupervised()
    X_train, X_test, y_train, y_test = _load_and_split_breast_cancer(True)

    for la in LA_RATIOS:
        idx_anomaly_train = np.where(y_train == 1)[0]
        n_train_anomalies = len(idx_anomaly_train)
        n_labeled = max(1, int(la * n_train_anomalies)) if n_train_anomalies > 0 else 0
        these_y_train = np.zeros_like(y_train)
        these_y_train[idx_anomaly_train[:n_labeled]] = 1

        semisup.fit(X_train, these_y_train)
        flextab.fit(X_train, these_y_train)
        semisup_scores = semisup.predict_score(X_test)
        flextab_scores = flextab.predict_proba(X_test)[:, 1]
        semisup_auroc = roc_auc_score(y_test, semisup_scores)
        flextab_auroc = roc_auc_score(y_test, flextab_scores)

        print(
            f'Semi-supervised la={la:.0%} ({n_labeled}/{n_train_anomalies} annotated): FlexTab AUROC = {flextab_auroc:.2%}, FlexTab-AD AUROC = {semisup_auroc:.2%}'
        )


def test_clustering():
    from sklearn.datasets import load_breast_cancer

    from flextab import FlexTabClusterer

    print('\nCLUSTERING')

    X, y = load_breast_cancer(return_X_y=True)
    n_clusters = len(np.unique(y))

    clusterer = FlexTabClusterer()
    pred = clusterer.fit(X).predict(X, n_clusters=n_clusters)

    acc = hungarian_accuracy(y, pred)
    ari = adjusted_rand_score(y, pred)
    print(f'Hungarian Accuracy: {acc:.2%}')
    print(f'ARI: {ari:.2%}')


def test_multi_table():
    from flextab import FlexTabRelationalEntityClassifier
    from flextab.flextab_multi.relbench_data import prepare_relbench_dataset

    print('\nMULTI-TABLE')

    rb_dataset = 'rel-f1'
    rb_task = 'driver-dnf'
    prepared_data = prepare_relbench_dataset(rb_dataset, rb_task, decoder_context_size=16384, query_size=-1)

    flextab_multi = FlexTabRelationalEntityClassifier()
    all_model_probs = flextab_multi.predict_proba(prepared_data)

    all_model_probs = all_model_probs[:, 1]  # probability of the positive class
    target_column = prepared_data.train_df.columns[-1]
    auroc = roc_auc_score(prepared_data.test_df[target_column], all_model_probs)
    print(f'\nFlexTab-Multi AUROC = {auroc:.1%} for Relbench dataset "{rb_dataset}" task "{rb_task}"')


def test_embedding():
    from sklearn.datasets import load_breast_cancer

    from flextab import FlexTabEmbedder

    print('\nEMBEDDING')

    X, _ = load_breast_cancer(return_X_y=True, as_frame=True)
    context = X.iloc[:30]
    query = X.iloc[30:50]

    embedder = FlexTabEmbedder(max_context_size=8192)
    expected_dim = embedder.model.config.hidden_size

    # Self-context: query=None -> (n_context, num_row_tokens * hidden)
    emb_context_only = embedder.embed(context)
    print(f'Self-context shape: {tuple(emb_context_only.shape)}')
    assert emb_context_only.shape == (len(context), expected_dim), (
        f'expected ({len(context)}, {expected_dim}), got {tuple(emb_context_only.shape)}'
    )

    # With explicit query: (n_context + n_query, num_row_tokens * hidden), context rows first
    emb_ctx_qry = embedder.embed(context, query)
    print(f'Context+query shape: {tuple(emb_ctx_qry.shape)}')
    assert emb_ctx_qry.shape == (len(context) + len(query), expected_dim), (
        f'expected ({len(context) + len(query)}, {expected_dim}), got {tuple(emb_ctx_qry.shape)}'
    )

    # Same context used both times: context-half of the explicit-query call must match the self-context call.
    same_context_diff = (emb_ctx_qry[: len(context)] - emb_context_only).abs().max().item()
    print(f'Context invariance max diff: {same_context_diff:.2e}')
    assert same_context_diff < 1e-3, (
        f'context embeddings must not change when a query is added, got max diff {same_context_diff}'
    )


if __name__ == '__main__':
    test_classification()
    test_regression()
    test_matching()
    test_embedding()
    test_unsupervised_anomaly_detection()
    test_oneclass_anomaly_detection()
    test_semisupervised_anomaly_detection()
    test_clustering()
    test_multi_table()
