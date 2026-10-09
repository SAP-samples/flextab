from dataclasses import dataclass

import numpy as np
import pandas as pd
import relbench
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from flextab.flextab_multi.bfs_discovery import get_entity_subgraph


@dataclass
class RelbenchData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    entity_table: str
    entity_col: str
    time_col: str
    train_subgraph: dict
    test_subgraph: dict
    train_entity_to_rows: dict
    test_entity_to_rows: dict
    train_features: dict
    test_features: dict
    pkey_cols: dict


def stringify_list_cells(subgraph: dict[str, pd.DataFrame]) -> None:
    """Stringify list/array cell values in all tables of a subgraph in-place.

    Some columns (e.g. product__category) contain numpy arrays or lists.
    These are converted to comma-separated strings so downstream encoders
    can treat them as categorical features.
    """
    for table_name, df in subgraph.items():
        for col in df.columns:
            if df[col].map(lambda x: isinstance(x, (list, np.ndarray))).any():
                df[col] = df[col].map(
                    lambda x: ', '.join(str(v) for v in x) if isinstance(x, (list, np.ndarray)) else x
                )


def expand_datetime_cols(subgraph: dict[str, pd.DataFrame]) -> None:
    """Replace datetime/date columns with day, month, year string columns in-place."""
    for table_name, df in subgraph.items():
        for col in list(df.columns):
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f'{col}_day'] = df[col].dt.day.astype(str)
                df[f'{col}_month'] = df[col].dt.month.astype(str)
                df[f'{col}_year'] = df[col].dt.year.astype(str)
                df.drop(columns=[col], inplace=True)


def merge_parents(
    db: relbench.base.Database,
    table_name: str,
    df: pd.DataFrame,
    entity_table_name: str,
    subgraph: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, set[str]]:
    """Left-join all FK leaf parent tables (except the entity table)
    Parent columns are prefixed with '<parent_name>__' to avoid collisions.
    """
    merged_into_child = set()
    tbl = db.table_dict[table_name]
    for fkey_col, parent_name in tbl.fkey_col_to_pkey_table.items():
        if parent_name == entity_table_name or parent_name not in subgraph:
            continue

        parent_tbl = db.table_dict[parent_name]
        # Only merge leaf parents (that contain no foreign keys))
        if parent_tbl.fkey_col_to_pkey_table:
            continue
        parent_df = subgraph[parent_name].set_index(parent_tbl.pkey_col).add_prefix(f'{parent_name}__')
        df = df.merge(parent_df, left_on=fkey_col, right_index=True, how='left')
        merged_into_child.add(parent_name)
    return df, merged_into_child


def drop_pk_fk(tbl: relbench.base.Table, df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Drop primary key and all foreign key columns from df."""
    cols_to_drop = set()
    if tbl.pkey_col and tbl.pkey_col in df.columns:
        cols_to_drop.add(tbl.pkey_col)
    for fkey_col in tbl.fkey_col_to_pkey_table:
        if fkey_col in df.columns:
            cols_to_drop.add(fkey_col)
    if entity_col in df.columns:
        cols_to_drop.add(entity_col)
    if '_entity_row_id' in df.columns:
        cols_to_drop.add('_entity_row_id')
    return df.drop(columns=list(cols_to_drop))


def get_feature_tables(
    subgraph: dict[str, pd.DataFrame],
    db: relbench.base.Database,
    entity_table_name: str,
    entity_col: str,
    merge_parents_flag: bool = True,
    is_train: bool = True,
) -> dict[str, pd.DataFrame]:
    """Get feature tables for embedding. Drop pk/fk from each table and optionally merge parent tables into children."""
    subgraph_features = {}
    merged_into_child = set()
    for table_name, df in subgraph.items():
        tbl = db.table_dict[table_name]
        if merge_parents_flag:
            df, merged = merge_parents(db, table_name, df, entity_table_name, subgraph)
            merged_into_child.update(merged)
        df = drop_pk_fk(tbl, df, entity_col)

        str_cols = [
            c
            for c in df.columns
            if 'id' in c.lower()
            or 'level' in c.lower()
            or 'year' in c.lower()
            or 'month' in c.lower()
            or 'day' in c.lower()
        ]
        df[str_cols] = df[str_cols].astype(str)

        if table_name == entity_table_name:
            subgraph_features[table_name] = df
        elif df.shape[1] > 1:
            # use tables with at least 2 columns
            subgraph_features[table_name] = df

    # Remove parent tables that are merged into their children
    if merge_parents_flag:
        for table_name in merged_into_child:
            subgraph_features.pop(table_name, None)

    set_type = 'Train' if is_train else 'Test'
    print(f'\n{set_type} feature tables to embed:')
    for table_name, df in subgraph_features.items():
        print(f'  {table_name}: {df.shape}, columns: {list(df.columns)}')
    return subgraph_features


def prepare_relbench_dataset(
    rb_dataset: str,
    rb_task: str,
    decoder_context_size: int = 16384,
    query_size: int = -1,
    n_hops: int = 2,
    max_bfs_width: int = 16,
) -> RelbenchData:
    ds = get_dataset(rb_dataset, download=True)
    db = ds.get_db()
    task = get_task(rb_dataset, rb_task, download=True)

    train_df = task.get_table('train').df
    test_df = task.get_table('test', mask_input_cols=False).df

    if query_size == -1:
        query_size = len(test_df)
    train_df_subset = train_df.sample(min(decoder_context_size, len(train_df)), random_state=42)
    test_df_subset = test_df.sample(min(query_size, len(test_df)), random_state=42)

    train_subgraph, train_entity_to_rows = get_entity_subgraph(
        db,
        entity_df=train_df_subset,
        entity_table_name=task.entity_table,
        entity_col=task.entity_col,
        time_col=task.time_col,
        last_n=max_bfs_width,
        n_hops=n_hops,
    )
    print('Discovered train tables:')
    for k in train_subgraph:
        print(f'  {k}: {train_subgraph[k].shape}')

    test_subgraph, test_entity_to_rows = get_entity_subgraph(
        db,
        entity_df=test_df_subset,
        entity_table_name=task.entity_table,
        entity_col=task.entity_col,
        time_col=task.time_col,
        last_n=max_bfs_width,
        n_hops=n_hops,
    )
    print('Discovered test tables:')
    for k in test_subgraph:
        print(f'  {k}: {test_subgraph[k].shape}')

    # format subgraph tables for encoding: stringify list cells, expand datetime cols, merge leaf parent tables into children and drop pk/fk
    stringify_list_cells(train_subgraph)
    stringify_list_cells(test_subgraph)
    expand_datetime_cols(train_subgraph)
    expand_datetime_cols(test_subgraph)

    train_subgraph_features = get_feature_tables(train_subgraph, db, task.entity_table, task.entity_col, is_train=True)
    test_subgraph_features = get_feature_tables(test_subgraph, db, task.entity_table, task.entity_col, is_train=False)

    # Build pkey column name mapping for each subgraph table
    pkey_cols = {}
    for table_name in train_subgraph:
        if '_entity_row_id' in train_subgraph[table_name].columns:
            pkey_cols[table_name] = '_entity_row_id'
        else:
            pkey_cols[table_name] = db.table_dict[table_name].pkey_col

    return RelbenchData(
        train_df=train_df_subset,
        test_df=test_df_subset,
        entity_table=task.entity_table,
        entity_col=task.entity_col,
        time_col=task.time_col,
        train_subgraph=train_subgraph,
        test_subgraph=test_subgraph,
        train_entity_to_rows=train_entity_to_rows,
        test_entity_to_rows=test_entity_to_rows,
        train_features=train_subgraph_features,
        test_features=test_subgraph_features,
        pkey_cols=pkey_cols,
    )
