"""BFS subgraph extraction for multi-table relational databases.

Extracts all rows reachable from a set of entity rows within n_hops,
traversing FK relationships in both directions:
  - Outgoing: child.fk -> parent  (e.g. review.product_id -> product)
  - Incoming: parent <- child.fk  (e.g. product <- review.product_id)

Key data structures maintained during BFS:
  collected:   {table_name: set of pkeys}          -- all rows reached so far
  reached_by:  {table_name: {pkey: {entity_ids}}}  -- which entities can reach each row
  frontier:    {table_name: set of pkeys}          -- newly reached rows
"""

from collections import defaultdict

import pandas as pd
import relbench


def apply_filters(
    matched_rows: pd.DataFrame,
    tbl: relbench.base.Table,
    cutoff: pd.Timestamp | None,
    has_timestamp: bool,
    last_n: int | None,
    fkey_col: str,
) -> pd.DataFrame:
    """Apply temporal cutoff and per-parent last_n to incoming child rows.

    Remove rows with timestamp > cutoff
    Per parent pkey (FK value), keep only the last_n most recent rows
        (or any last_n rows if the table has no time column)

    Returns:
      filtered_rows
    """
    has_time = tbl.time_col and tbl.time_col in matched_rows.columns

    if has_time and has_timestamp and cutoff is not None:
        matched_rows = matched_rows[matched_rows[tbl.time_col] <= cutoff]

    if len(matched_rows) == 0 or last_n is None:
        return matched_rows

    if has_time:
        # Per parent pkey, keep last_n most recent child rows
        matched_rows = matched_rows.sort_values(tbl.time_col).groupby(fkey_col, sort=False).tail(last_n)
    else:
        # No time column: keep any last_n rows per parent pkey (unordered)
        matched_rows = matched_rows.groupby(fkey_col, sort=False).tail(last_n)

    return matched_rows


def ensure_pkeys(db: relbench.base.Database) -> relbench.base.Database:
    """Add a synthetic integer pkey to any table that lacks one."""
    for table_name, tbl in db.table_dict.items():
        if tbl.pkey_col is None:
            tbl.df = tbl.df.copy()
            tbl.df.insert(0, 'primary_key', tbl.df.index)
            tbl.pkey_col = 'primary_key'
    return db


def _run_bfs(
    db: relbench.base.Database,
    entity_df: pd.DataFrame,
    entity_table_name: str,
    entity_col: str,
    time_col: str,
    n_hops: int,
    last_n: int | None,
    is_print: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Core BFS over the database graph starting from a set of entity rows.

    Assumptions (enforced by the caller get_entity_subgraph):
      - All rows in entity_df share the same timestamp (or there is no timestamp).
        The cutoff is read from iloc[0]; mixing timestamps would silently use
        the wrong cutoff for most entities.
      - entity_df contains no duplicate entity IDs (asserted below).

    Returns:
      result         - {table_name: DataFrame} with all rows reachable within n_hops
      entity_to_rows - {table_name: {entity_id: [pkeys]}} mapping each entity to
                       the rows it "owns" in each table
    """
    has_timestamp = time_col in entity_df.columns
    unique_entity_ids = set(int(x) for x in entity_df[entity_col].values)
    assert len(unique_entity_ids) == len(entity_df), 'entity_df must not contain duplicate entity IDs'
    # All rows share the same cutoff (guaranteed by get_entity_subgraph splitting by timestamp)
    cutoff = entity_df[time_col].iloc[0] if has_timestamp else None

    # -- BFS state --
    # collected[table_name]  = set of all pkeys reached in that table (across all hops)
    # reached_by[table][pkey] = set of entity_ids that can reach that row
    # reached_by: {table_name: {pkey: {entity_ids}}}  — which entities reached it
    collected = defaultdict(set)
    reached_by = defaultdict(lambda: defaultdict(set))

    # Seed BFS with the entity rows themselves
    collected[entity_table_name] = set(unique_entity_ids)
    for entity_id in unique_entity_ids:
        reached_by[entity_table_name][entity_id].add(entity_id)

    frontier = {entity_table_name: set(unique_entity_ids)}

    # Pre-compute reverse connections: parent_table -> [(child_table, fkey_col)]
    # Used to find all child tables that reference a given parent during incoming traversal.
    # For example, product -> [(review, product_id)] since review.product_id references product.
    parent_to_children = defaultdict(list)
    for child_name, child_tbl in db.table_dict.items():
        for fkey_col, parent_name in child_tbl.fkey_col_to_pkey_table.items():
            parent_to_children[parent_name].append((child_name, fkey_col))

    # -- BFS loop --
    for hop in range(n_hops):
        next_frontier = defaultdict(set)

        for table_name, new_pkeys in frontier.items():
            if not new_pkeys:
                continue
            tbl = db.table_dict[table_name]
            current_rows = tbl.df[tbl.df[tbl.pkey_col].astype(int).isin(new_pkeys)]

            # ── Outgoing edges: follow FK -> parent table
            # e.g. review.product_id -> product
            # No temporal filter here: parent tables (product, customer) are lookup tables without timestamps.
            for fkey_col, parent_name in tbl.fkey_col_to_pkey_table.items():
                valid = current_rows[current_rows[fkey_col].notna()]
                primary_keys = valid[tbl.pkey_col].astype(int).values
                foreign_keys = valid[fkey_col].astype(int).values

                # Propagate ownership through the FK edge
                pkey_to_entities = defaultdict(set)
                for pk, fk in zip(primary_keys, foreign_keys):
                    pkey_to_entities[int(fk)].update(reached_by[table_name][int(pk)])

                for parent_pkey in set(int(x) for x in foreign_keys):
                    reached_by[parent_name][parent_pkey].update(pkey_to_entities[parent_pkey])
                    if parent_pkey not in collected[parent_name]:
                        collected[parent_name].add(parent_pkey)
                        next_frontier[parent_name].add(parent_pkey)

            # ── Incoming edges: find child rows pointing at current table
            # e.g. product <- review.product_id
            # Temporal filter is applied here to prevent using future child rows
            for child_name, fkey_col in parent_to_children[table_name]:
                child_tbl = db.table_dict[child_name]
                child_df = child_tbl.df

                # Build boolean mask: child rows whose FK is in new_pkeys
                mask = child_df[fkey_col].dropna().astype(int).isin(new_pkeys)
                # dropna() changes the index, so reindex back to full child_df index
                mask = mask.reindex(child_df.index, fill_value=False)
                matched_rows = child_df[mask].copy()

                if len(matched_rows) == 0:
                    continue

                # Apply temporal cutoff and last_n trimming
                matched_rows = apply_filters(matched_rows, child_tbl, cutoff, has_timestamp, last_n, fkey_col)

                # Map each parent pkey to the entity_ids that reached it
                pkey_to_entities = defaultdict(set)
                for pkey in new_pkeys:
                    pkey_to_entities[pkey].update(reached_by[table_name][pkey])

                # Propagate entity ownership from parent to newly reached child rows
                child_primary_keys = matched_rows[child_tbl.pkey_col].astype(int).values
                parent_primary_keys = matched_rows[fkey_col].astype(int).values
                for child_pkey, parent_pkey in zip(child_primary_keys, parent_primary_keys):
                    child_pkey, parent_pkey = int(child_pkey), int(parent_pkey)
                    reached_by[child_name][child_pkey].update(pkey_to_entities[parent_pkey])
                    if child_pkey not in collected[child_name]:
                        collected[child_name].add(child_pkey)
                        next_frontier[child_name].add(child_pkey)

        frontier = dict(next_frontier)
        if is_print:
            total = sum(len(v) for v in frontier.values())
            tables = list(frontier.keys())
            print(f'  hop {hop + 1}: {total:,} new rows across {len(tables)} from tables {tables}')

    # Build result dfs using pkey values directly
    result = {}
    for table_name, pkeys in collected.items():
        if not pkeys:
            continue
        tbl = db.table_dict[table_name]
        result[table_name] = tbl.df[tbl.df[tbl.pkey_col].astype(int).isin(pkeys)].copy()
        if is_print:
            print(f'  {table_name}: {len(result[table_name]):,} rows')

    # entity_to_rows: {table_name: {entity_id: [pkey, ...]}}
    # for example:
    # "product":  {5: [5]}        # entity 5 -> product row 5
    # "review":   {5: [1, 3]}     # entity 5 -> review rows 1 and 3
    # "customer": {5: [42, 99]}   # entity 5 -> customer rows 42 and 99
    # just reverse reached_by to entity_to_rows
    # to have the mapping from entity to rows it can reaches, needed for per-entity embedding gathering later
    entity_to_rows = {}
    for table_name, pkeys in collected.items():
        mapping = defaultdict(list)
        for pkey, entity_ids in reached_by[table_name].items():
            if pkey in pkeys:
                for entity_id in entity_ids:
                    mapping[entity_id].append(pkey)
        entity_to_rows[table_name] = dict(mapping)

    return result, entity_to_rows


def get_entity_subgraph(
    db: relbench.base.Database,
    entity_df: pd.DataFrame,
    entity_table_name: str,
    entity_col: str,
    time_col: str,
    n_hops: int = 2,
    last_n: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Extract all rows reachable from entity_df within n_hops in the database graph.

    Handles multiple unique timestamps by running one BFS per unique timestamp
    and unioning results.  This is necessary because the temporal cutoff differs
    per task row when the same entity appears at multiple timestamps.

    Example entity_df:
        product_id  timestamp
        5           2020-06-01   <- task row A  (cutoff: 2020-06-01)
        5           2021-01-01   <- task row B  (same product, later cutoff)
        9           2020-06-01   <- task row C

    Returns:
        result          — {table_name: DataFrame} union of all BFS passes
        entity_to_rows  — {table_name: {(entity_id, timestamp): [pkeys]}}
                          keyed by (entity_id, timestamp) so task row A and B
                          for the same product get separate row sets
    """
    ensure_pkeys(db)

    has_timestamp = time_col in entity_df.columns
    # Pre-merge entity_df with entity table columns so the entity table in the
    # result has one row per (entity_id, timestamp), matching entity_df exactly.
    entity_tbl = db.table_dict[entity_table_name]
    original_pkey_col = entity_tbl.pkey_col
    if has_timestamp:
        entity_merged = entity_df[[entity_col, time_col]].merge(
            entity_tbl.df, left_on=entity_col, right_on=original_pkey_col, how='left'
        )
        entity_merged['_entity_row_id'] = range(len(entity_merged))

        # Handle timestamp column name clash: if entity table has a column
        # with the same name as time_col, the merge creates {time_col}_x
        # (from entity_df) and {time_col}_y (from entity table).
        tc_x, tc_y = f'{time_col}_x', f'{time_col}_y'
        if tc_x in entity_merged.columns:
            entity_merged = entity_merged.drop(columns=[tc_x]).rename(columns={tc_y: time_col})

    if not has_timestamp:
        result, entity_to_rows = _run_bfs(db, entity_df, entity_table_name, entity_col, time_col, n_hops, last_n)
        return result, entity_to_rows

    unique_timestamps = entity_df[time_col].unique()

    def _finalize_entity_table(result, etr):
        """Replace entity table with entity_merged + any BFS context rows.

        entity_merged provides one row per (entity_id, timestamp)
        BFS may discover additional entity-table rows
        """
        # Map each (entity_id, timestamp) -> its position in entity_merged
        entity_mapping = {}
        for rid, (eid, ts) in enumerate(zip(entity_df[entity_col].values, entity_df[time_col].values)):
            entity_mapping[(int(eid), pd.Timestamp(ts))] = [rid]

        # Find BFS rows not in entity_df (context from BFS)
        bfs_df = result.get(entity_table_name, pd.DataFrame())
        entity_id_set = set(entity_merged[entity_col].astype(int).values)
        if len(bfs_df) > 0 and original_pkey_col in bfs_df.columns:
            extra = bfs_df[~bfs_df[original_pkey_col].astype(int).isin(entity_id_set)].copy()
        else:
            extra = pd.DataFrame()

        if len(extra) > 0:
            n = len(entity_merged)
            extra['_entity_row_id'] = range(n, n + len(extra))
            combined = pd.concat([entity_merged, extra], ignore_index=True)
            # Attach extra rows to the entities that discovered them
            pk_to_rid = {int(pk): n + i for i, pk in enumerate(extra[original_pkey_col].astype(int).values)}
            for (eid, ts), pkeys in etr.get(entity_table_name, {}).items():
                rids = [pk_to_rid[p] for p in pkeys if p in pk_to_rid]
                if rids:
                    entity_mapping.setdefault((int(eid), pd.Timestamp(ts)), []).extend(rids)
        else:
            combined = entity_merged

        result[entity_table_name] = combined
        etr[entity_table_name] = entity_mapping
        return result, etr

    if len(unique_timestamps) == 1:
        result, entity_to_rows = _run_bfs(db, entity_df, entity_table_name, entity_col, time_col, n_hops, last_n)
        timestamp = pd.Timestamp(unique_timestamps[0])
        # Re-key entity_to_rows to use (entity_id, timestamp) key for consistency with multi-timestamp case
        entity_to_rows = {
            table_name: {(entity_id, timestamp): pkeys for entity_id, pkeys in mapping.items()}
            for table_name, mapping in entity_to_rows.items()
        }
        result, etr = _finalize_entity_table(result, entity_to_rows)
        return result, etr

    # `all_results` accumulates all pkeys ever reached across all BFS passes.
    # Stored as {pkey: None} (dict instead of set) so we can call .keys() later.
    # {table_name: {pkey: None, pkey: None, ...}}
    # For example:
    # "product":  {5: None, 9: None}
    # "review":   {1: None, 3: None}
    # "customer": {42: None, 99: None}
    all_results = defaultdict(dict)

    # `all_entity_to_rows` accumulates the entity -> rows mapping across all passes.
    # Key is (entity_id, timestamp) to keep separate row sets per task row.
    # {table_name: {(entity_id, timestamp): [pkeys]}}
    # For example:
    # "review": {
    #     (5, 2020-06-01): [1, 3],      # task row A — only sees reviews before 2020
    #     (5, 2021-01-01): [1, 3, 7],   # task row B — sees one more review
    # }
    all_entity_to_rows = defaultdict(lambda: defaultdict(list))

    for timestamp in sorted(unique_timestamps):
        # Run BFS for all entities at this timestamp (same cutoff for all rows in this group)
        group = entity_df[entity_df[time_col] == timestamp]
        result, entity_to_rows = _run_bfs(db, group, entity_table_name, entity_col, time_col, n_hops, last_n)
        # Collect all reached pkeys into all_results (union across passes)
        for table_name, df in result.items():
            tbl = db.table_dict[table_name]
            for pkey in df[tbl.pkey_col].astype(int).values:
                all_results[table_name][pkey] = None

        # Store per-entity row lists under the (entity_id, timestamp) composite key
        for table_name, mapping in entity_to_rows.items():
            for entity_id, pkeys in mapping.items():
                composite_key = (entity_id, pd.Timestamp(timestamp))
                all_entity_to_rows[table_name][composite_key].extend(pkeys)

    # Deduplicate pkeys per (entity_id, timestamp)
    for table_name in all_entity_to_rows:
        for composite_key in all_entity_to_rows[table_name]:
            all_entity_to_rows[table_name][composite_key] = list(set(all_entity_to_rows[table_name][composite_key]))

    # Build final DataFrames from the union of all reached pkeys
    final_result = {}
    for table_name, pkey_dict in all_results.items():
        tbl = db.table_dict[table_name]
        final_result[table_name] = tbl.df[tbl.df[tbl.pkey_col].astype(int).isin(pkey_dict.keys())].copy()

    final_entity_to_rows = {table_name: dict(mapping) for table_name, mapping in all_entity_to_rows.items()}
    result, etr = _finalize_entity_table(final_result, final_entity_to_rows)
    return result, etr
