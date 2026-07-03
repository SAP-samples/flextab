from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from flextab.utils.test_utils import _load_and_split_breast_cancer, load_febrl4


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
