#!/usr/bin/env python3
"""
Migration script to move data from SQLite to PostgreSQL
"""
import os
import sys
import sqlite3
import psycopg2
import pandas as pd
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logger

class SQLiteToPostgreSQLMigrator:
    def __init__(self, sqlite_path: str, postgres_url: str):
        self.sqlite_path = sqlite_path
        self.postgres_url = postgres_url
        self.logger = setup_logger("DatabaseMigrator")
        
    def migrate_all_data(self):
        try:
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row
            pg_conn = psycopg2.connect(self.postgres_url)
            pg_conn.autocommit = True

            cur = sqlite_conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cur.fetchall()]
            self.logger.info(f"Found tables: {tables}")

            for table in tables:
                self.migrate_table(sqlite_conn, pg_conn, table)

            sqlite_conn.close()
            pg_conn.close()
            self.logger.info("Migration completed successfully")
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise

    def migrate_table(self, sqlite_conn: sqlite3.Connection, pg_conn: psycopg2.extensions.connection, table: str):
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", sqlite_conn)
            if df.empty:
                self.logger.info(f"Table {table} is empty; skipping")
                return
            # Basic create table
            cursor = pg_conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            col_defs = []
            for col, dtype in df.dtypes.items():
                if dtype == 'object':
                    t = 'TEXT'
                elif dtype == 'int64':
                    t = 'INTEGER'
                elif dtype == 'float64':
                    t = 'REAL'
                elif 'datetime' in str(dtype):
                    t = 'TIMESTAMP'
                elif dtype == 'bool':
                    t = 'BOOLEAN'
                else:
                    t = 'TEXT'
                col_defs.append(f"{col} {t}")
            cursor.execute(f"CREATE TABLE {table} ({', '.join(col_defs)})")

            from sqlalchemy import create_engine
            engine = create_engine(self.postgres_url)
            df.to_sql(table, engine, if_exists='replace', index=False, method='multi')
            self.logger.info(f"Migrated {len(df)} rows to {table}")
        except Exception as e:
            self.logger.error(f"Failed to migrate table {table}: {e}")
            raise


def main():
    sqlite_path = os.getenv("SQLITE_PATH", "data/customer_insights.db")
    postgres_url = os.getenv("DATABASE_URL", "postgresql://cicop_user:cicop_pass@localhost:5432/cicop_db")
    print(f"Starting migration from {sqlite_path} to {postgres_url}")
    migrator = SQLiteToPostgreSQLMigrator(sqlite_path, postgres_url)
    migrator.migrate_all_data()

if __name__ == "__main__":
    main()
