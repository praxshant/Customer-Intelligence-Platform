# database_manager.py
# Purpose: Manage SQLite database operations and connections

import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import os
from contextlib import contextmanager
from src.utils.logger import setup_logger
import os
import re
try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None  # Optional if not using Postgres
try:
    from sqlalchemy import create_engine
except Exception:
    create_engine = None

class DatabaseManager:
    def __init__(self, db_path: str = "data/customer_insights.db"):
        # Step 1: Initialize database configuration
        self.logger = setup_logger("DatabaseManager")
        env_url = os.getenv("DATABASE_URL", "")
        # Accept postgres:// or postgresql:// or sqlite schemas; fallback to provided db_path
        if env_url:
            self.database_url = env_url
        else:
            # Normalize sqlite path to URL form for consistency
            if db_path == ":memory:":
                self.database_url = "sqlite:///:memory:"
            else:
                self.database_url = f"sqlite:///{db_path}"
        self.db_type = self._determine_db_type()

        # Backward-compat: expose db_path and persistent connection for prior code paths
        self.db_path = db_path if not env_url else (db_path if self.db_type == "sqlite" else db_path)

        # Create directory for SQLite file DBs
        if self.db_type == "sqlite":
            real_path = self.database_url.replace("sqlite:///", "").replace("sqlite://", "")
            db_dir = os.path.dirname(real_path)
            if real_path not in (":memory:", "") and db_dir:
                os.makedirs(db_dir, exist_ok=True)

        # Maintain a persistent connection for in-memory SQLite databases only
        self._persistent_conn = sqlite3.connect(":memory:") if self.database_url.endswith(":memory:") else None

        # Step 2: Initialize database schema
        self._initialize_database()

    def _determine_db_type(self) -> str:
        url = self.database_url.lower()
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            return "postgresql"
        return "sqlite"

    @contextmanager
    def _get_connection(self):
        """Context manager that yields a DB-API connection for the configured backend."""
        conn = None
        try:
            if self.db_type == "postgresql":
                if psycopg2 is None:
                    raise RuntimeError("psycopg2 is required for PostgreSQL but not installed")
                conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            else:
                if self._persistent_conn is not None:
                    conn = self._persistent_conn
                else:
                    real_path = self.database_url.replace("sqlite:///", "").replace("sqlite://", "")
                    conn = sqlite3.connect(real_path, timeout=30, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
            yield conn
        finally:
            # Do not close persistent in-memory connection
            if conn is not None and conn is not self._persistent_conn and self.db_type == "sqlite":
                conn.close()
            if conn is not None and self.db_type == "postgresql":
                conn.close()
    
    def _initialize_database(self):
        # Step 1: Create database tables if they don't exist
        if self._persistent_conn is not None:
            conn = self._persistent_conn
            cursor = conn.cursor()
        else:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create customers table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS customers (
                        customer_id TEXT PRIMARY KEY,
                        first_name TEXT,
                        last_name TEXT,
                        email TEXT,
                        registration_date DATE,
                        age INTEGER,
                        gender TEXT,
                        location TEXT
                    )
                """)
                
                # Create transactions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        transaction_id TEXT PRIMARY KEY,
                        customer_id TEXT,
                        transaction_date DATE,
                        amount REAL,
                        category TEXT,
                        subcategory TEXT,
                        merchant TEXT,
                        payment_method TEXT,
                        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                    )
                """)
                
                # Create customer_segments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS customer_segments (
                        customer_id TEXT PRIMARY KEY,
                        segment_id INTEGER,
                        segment_name TEXT,
                        last_updated DATE,
                        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                    )
                """)
                
                # Create campaigns table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS campaigns (
                        campaign_id TEXT PRIMARY KEY,
                        customer_id TEXT,
                        campaign_type TEXT,
                        offer_description TEXT,
                        discount_amount REAL,
                        created_date DATE,
                        status TEXT,
                        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_location ON customers(location)")
                
                conn.commit()
                return
        # If using persistent connection path, run schema creation and commit
        if self._persistent_conn is not None:
            # Create customers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    first_name TEXT,
                    last_name TEXT,
                    email TEXT,
                    registration_date DATE,
                    age INTEGER,
                    gender TEXT,
                    location TEXT
                )
            """)
            # Create transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    transaction_date DATE,
                    amount REAL,
                    category TEXT,
                    subcategory TEXT,
                    merchant TEXT,
                    payment_method TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
            """)
            # Create customer_segments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_segments (
                    customer_id TEXT PRIMARY KEY,
                    segment_id INTEGER,
                    segment_name TEXT,
                    last_updated DATE,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
            """)
            # Create campaigns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    campaign_type TEXT,
                    offer_description TEXT,
                    discount_amount REAL,
                    created_date DATE,
                    status TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
            """)
            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_location ON customers(location)")
            conn.commit()
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        # Step 1: Insert pandas DataFrame into database table
        if self.db_type == "postgresql":
            if create_engine is None:
                raise RuntimeError("SQLAlchemy is required for PostgreSQL DataFrame insertion")
            engine = create_engine(self.database_url)
            df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi')
            self.logger.info(f"Inserted {len(df)} records into {table_name}")
            return
        with (self._persistent_conn if self._persistent_conn else sqlite3.connect(self.db_path)) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            self.logger.info(f"Inserted {len(df)} records into {table_name}")
    
    def query_to_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        # Step 1: Execute SQL query and return as DataFrame
        if self.db_type == "postgresql":
            with self._get_connection() as conn:
                return pd.read_sql_query(query.replace('?', '%s'), conn, params=params)
        with (self._persistent_conn if self._persistent_conn else sqlite3.connect(self.db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            return df
    
    def execute_query(self, query: str, params: Optional[tuple] = None):
        # Step 1: Execute SQL query without returning results
        if self.db_type == "postgresql":
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query.replace('?', '%s'), params or ())
                conn.commit()
                # Try fetch if it's a SELECT
                if query.strip().lower().startswith(('select', 'with')):
                    rows = cursor.fetchall()
                    return rows
                return cursor.rowcount
        with (self._persistent_conn if self._persistent_conn else sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            return cursor.rowcount
    
    def get_customer_data(self, customer_id: Optional[str] = None) -> pd.DataFrame:
        # Step 1: Retrieve customer data with optional filtering
        query = """
            SELECT c.*, cs.segment_name
            FROM customers c
            LEFT JOIN customer_segments cs ON c.customer_id = cs.customer_id
        """
        
        if customer_id:
            query += " WHERE c.customer_id = ?"
            return self.query_to_dataframe(query, (customer_id,))
        
        return self.query_to_dataframe(query)
    
    def get_transaction_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        # Step 1: Retrieve transaction data with optional date filtering
        query = """
            SELECT t.*, c.first_name, c.last_name, cs.segment_name
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            LEFT JOIN customer_segments cs ON t.customer_id = cs.customer_id
        """
        
        params = []
        if start_date and end_date:
            query += " WHERE t.transaction_date BETWEEN ? AND ?"
            params = [start_date, end_date]
        
        return self.query_to_dataframe(query, tuple(params) if params else None)
    
    def get_campaign_data(self) -> pd.DataFrame:
        # Step 1: Retrieve campaign data
        query = """
            SELECT c.*, cu.first_name, cu.last_name, cu.email
            FROM campaigns c
            JOIN customers cu ON c.customer_id = cu.customer_id
            ORDER BY c.created_date DESC
        """
        return self.query_to_dataframe(query)
    
    def update_customer_segments(self, segments_df: pd.DataFrame):
        # Step 1: Update customer segments
        self.insert_dataframe(segments_df, "customer_segments", if_exists="replace")
        self.logger.info(f"Updated segments for {len(segments_df)} customers")
    
    def get_database_stats(self) -> Dict[str, int]:
        # Step 1: Get database statistics
        stats = {}
        tables = ['customers', 'transactions', 'customer_segments', 'campaigns']
        
        for table in tables:
            query = f"SELECT COUNT(*) FROM {table}"
            count = self.query_to_dataframe(query).iloc[0, 0]
            # Cast to native Python int to prevent numpy.int64 in API responses
            try:
                stats[table] = int(count)
            except Exception:
                stats[table] = 0
        
        return stats 