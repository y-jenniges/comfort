"""
Module containing functions to communicate with the database.
"""
import logging
import sqlite3


def create_connection(db_file):
    """
    Create a database connection to the SQLite database db_file.

    Args:
        db_file (str): Path to database file.
    Returns:
        sqlite3.Connection: Connection object or None.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        logging.error(f"Error: Could not establish a connection to {db_file}")
        logging.error(e)
    return conn

