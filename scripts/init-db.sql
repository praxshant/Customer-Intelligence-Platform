-- scripts/init-db.sql
-- Initialize PostgreSQL schema to mirror SQLite structures used by the app

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    registration_date DATE,
    age INTEGER,
    gender TEXT,
    location TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(customer_id),
    transaction_date DATE,
    amount REAL,
    category TEXT,
    subcategory TEXT,
    merchant TEXT,
    payment_method TEXT
);

CREATE TABLE IF NOT EXISTS customer_segments (
    customer_id TEXT PRIMARY KEY REFERENCES customers(customer_id),
    segment_id INTEGER,
    segment_name TEXT,
    last_updated DATE
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(customer_id),
    campaign_type TEXT,
    offer_description TEXT,
    discount_amount REAL,
    created_date DATE,
    status TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_location ON customers(location);
