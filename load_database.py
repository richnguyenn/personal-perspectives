"""Load comments from a SQLite database.

This module provides a function to load Reddit comments from a SQLite database
for use in personality prediction. It uses only Python's built-in sqlite3 library.
"""

import sqlite3


def load_comments(db_path, table_name="comments", text_col="body", limit=None, where_clause=None):
    """Load comments from a SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    table_name : str, optional
        Name of the table to query (default: "comments").
    text_col : str, optional
        Name of the column containing the text to analyze (default: "body").
    limit : int, optional
        Maximum number of rows to load. If None, load all rows.
    where_clause : str, optional
        Optional WHERE clause for filtering (e.g., "subreddit = 'python'").
        Do not include the word "WHERE" in the clause.

    Returns
    -------
    list of dict
        List of comment dictionaries. Each dict has keys for each column
        in the table (e.g., id, author, body, subreddit).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build the query
    query = "SELECT * FROM " + table_name
    if where_clause:
        query = query + " WHERE " + where_clause
    if limit is not None:
        query = query + " LIMIT " + str(limit)

    cursor.execute(query)
    rows = cursor.fetchall()

    # Convert each row to a dictionary
    comments = []
    for row in rows:
        row_dict = dict(row)
        comments.append(row_dict)

    conn.close()
    return comments
