#!/usr/bin/env python3
# generate_standalone_data.py
# Purpose: Generate comprehensive sample data using only built-in Python libraries

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class StandaloneDataGenerator:
    def __init__(self, db_path="data/customer_insights.db"):
        # Step 1: Initialize generator
        self.db_path = db_path
        self.ensure_data_directory()
        self.setup_database()
        
        # Step 2: Define realistic business categories and merchants
        self.categories = {
            'Food & Dining': [
                'McDonald\'s', 'Starbucks', 'Subway', 'Pizza Hut', 'KFC',
                'Burger King', 'Domino\'s', 'Chipotle', 'Panera Bread', 'Dunkin\''
            ],
            'Shopping': [
                'Amazon', 'Walmart', 'Target', 'Best Buy', 'Home Depot',
                'Costco', 'Macy\'s', 'Kohl\'s', 'Nordstrom', 'Sephora'
            ],
            'Entertainment': [
                'Netflix', 'Spotify', 'Disney+', 'Hulu', 'HBO Max',
                'AMC Theaters', 'Regal Cinemas', 'GameStop', 'Steam', 'PlayStation'
            ],
            'Transportation': [
                'Uber', 'Lyft', 'Shell', 'ExxonMobil', 'BP',
                'Chevron', 'Hertz', 'Enterprise', 'Delta Airlines', 'Southwest'
            ],
            'Health & Wellness': [
                'CVS Pharmacy', 'Walgreens', 'Gym Membership', 'Yoga Studio',
                'Vitamin Shop', 'Medical Center', 'Dental Office', 'Optometry'
            ],
            'Technology': [
                'Apple Store', 'Microsoft Store', 'Google Play', 'Adobe',
                'Zoom', 'Slack', 'Dropbox', 'GitHub', 'AWS', 'DigitalOcean'
            ],
            'Travel': [
                'Booking.com', 'Expedia', 'Airbnb', 'Marriott', 'Hilton',
                'American Airlines', 'United Airlines', 'Southwest', 'Cruise Lines'
            ],
            'Education': [
                'Coursera', 'Udemy', 'edX', 'Khan Academy', 'University',
                'Online Course', 'Textbook Store', 'Language Learning'
            ]
        }
        
        self.payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay', 'Cash']
        self.first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa', 'James', 'Jennifer',
                           'William', 'Linda', 'Richard', 'Patricia', 'Joseph', 'Barbara', 'Thomas', 'Elizabeth', 'Christopher', 'Jessica',
                           'Charles', 'Sarah', 'Daniel', 'Karen', 'Matthew', 'Nancy', 'Anthony', 'Betty', 'Mark', 'Helen',
                           'Donald', 'Sandra', 'Steven', 'Donna', 'Paul', 'Carol', 'Andrew', 'Ruth', 'Joshua', 'Sharon',
                           'Kenneth', 'Michelle', 'Kevin', 'Laura', 'Brian', 'Emily', 'George', 'Kimberly', 'Edward', 'Deborah']
        self.last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                          'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                          'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson',
                          'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
                          'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter', 'Roberts']
        self.cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                      'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington',
                      'Boston', 'El Paso', 'Nashville', 'Detroit', 'Oklahoma City', 'Portland', 'Las Vegas', 'Memphis', 'Louisville', 'Baltimore']
        self.states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'NC', 'WA', 'GA', 'MI', 'CO', 'TN', 'OR', 'NV', 'KY', 'MD', 'MO', 'WI']
    
    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def setup_database(self):
        """Setup SQLite database with all required tables"""
        print("Setting up database...")
        
        with sqlite3.connect(self.db_path) as conn:
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
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_location ON customers(location)")
            
            conn.commit()
        
        print("Database setup completed!")
    
    def generate_customers(self, num_customers: int = 1000) -> pd.DataFrame:
        """Generate customer data"""
        print(f"Generating {num_customers} customers...")
        
        customers = []
        for i in range(num_customers):
            customer = {
                'customer_id': f'CUST_{i+1:06d}',
                'first_name': random.choice(self.first_names),
                'last_name': random.choice(self.last_names),
                'email': f'customer{i+1}@example.com',
                'registration_date': (datetime.now() - timedelta(days=random.randint(1, 730))).strftime('%Y-%m-%d'),
                'age': random.randint(18, 80),
                'gender': random.choice(['Male', 'Female', 'Other']),
                'location': f"{random.choice(self.cities)}, {random.choice(self.states)}"
            }
            customers.append(customer)
        
        customers_df = pd.DataFrame(customers)
        print(f"Generated {len(customers_df)} customers")
        return customers_df
    
    def generate_transactions(self, customers_df: pd.DataFrame, 
                            avg_transactions_per_customer: int = 15) -> pd.DataFrame:
        """Generate transaction data"""
        total_transactions = len(customers_df) * avg_transactions_per_customer
        print(f"Generating {total_transactions} transactions...")
        
        transactions = []
        transaction_id = 1
        
        for _, customer in customers_df.iterrows():
            num_transactions = random.randint(1, avg_transactions_per_customer * 2)
            
            start_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
            end_date = datetime.now()
            
            for _ in range(num_transactions):
                days_between = (end_date - start_date).days
                random_days = random.randint(0, days_between)
                transaction_date = (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
                
                category = random.choice(list(self.categories.keys()))
                merchant = random.choice(self.categories[category])
                amount = self._generate_realistic_amount(category)
                
                transaction = {
                    'transaction_id': f'TXN_{transaction_id:08d}',
                    'customer_id': customer['customer_id'],
                    'transaction_date': transaction_date,
                    'amount': amount,
                    'category': category,
                    'subcategory': self._generate_subcategory(category),
                    'merchant': merchant,
                    'payment_method': random.choice(self.payment_methods)
                }
                transactions.append(transaction)
                transaction_id += 1
        
        transactions_df = pd.DataFrame(transactions)
        transactions_df = transactions_df.sort_values('transaction_date')
        transactions_df = transactions_df.drop_duplicates(subset=['transaction_id'])
        
        print(f"Generated {len(transactions_df)} transactions")
        return transactions_df
    
    def _generate_realistic_amount(self, category: str) -> float:
        """Generate realistic amounts based on category"""
        amount_ranges = {
            'Food & Dining': (5, 50),
            'Shopping': (20, 500),
            'Entertainment': (10, 200),
            'Transportation': (15, 100),
            'Health & Wellness': (10, 300),
            'Technology': (50, 2000),
            'Travel': (100, 5000),
            'Education': (20, 1000)
        }
        
        min_amount, max_amount = amount_ranges.get(category, (10, 100))
        mu = np.log((min_amount + max_amount) / 2)
        sigma = 0.5
        
        amount = np.random.lognormal(mu, sigma)
        amount = max(min_amount, min(max_amount, amount))
        
        return round(amount, 2)
    
    def _generate_subcategory(self, category: str) -> str:
        """Generate realistic subcategories"""
        subcategories = {
            'Food & Dining': ['Fast Food', 'Coffee', 'Restaurant', 'Delivery', 'Catering'],
            'Shopping': ['Clothing', 'Electronics', 'Home & Garden', 'Books', 'Sports'],
            'Entertainment': ['Streaming', 'Movies', 'Games', 'Music', 'Events'],
            'Transportation': ['Ride Share', 'Gas', 'Car Rental', 'Public Transit', 'Flights'],
            'Health & Wellness': ['Pharmacy', 'Fitness', 'Medical', 'Dental', 'Supplements'],
            'Technology': ['Software', 'Hardware', 'Services', 'Apps', 'Cloud'],
            'Travel': ['Hotels', 'Flights', 'Vacation Rentals', 'Car Rentals', 'Activities'],
            'Education': ['Online Courses', 'Books', 'Software', 'Certifications', 'Workshops']
        }
        
        return random.choice(subcategories.get(category, ['General']))
    
    def generate_seasonal_patterns(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal patterns to transactions"""
        print("Adding seasonal patterns to transactions...")
        
        seasonal_multipliers = {
            12: 1.3,  # December - Holiday season
            1: 0.8,   # January - Post-holiday slump
            2: 0.9,   # February - Valentine's
            3: 1.0,   # March - Spring
            4: 1.1,   # April - Spring
            5: 1.0,   # May - Spring
            6: 1.2,   # June - Summer
            7: 1.3,   # July - Summer vacation
            8: 1.2,   # August - Summer
            9: 1.0,   # September - Back to school
            10: 1.1,  # October - Halloween
            11: 1.2   # November - Thanksgiving
        }
        
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        transactions_df['month'] = transactions_df['transaction_date'].dt.month
        
        for month, multiplier in seasonal_multipliers.items():
            mask = transactions_df['month'] == month
            transactions_df.loc[mask, 'amount'] = transactions_df.loc[mask, 'amount'] * multiplier
        
        transactions_df['amount'] = transactions_df['amount'].round(2)
        transactions_df = transactions_df.drop('month', axis=1)
        
        print("Seasonal patterns applied successfully")
        return transactions_df
    
    def generate_customer_segments(self, customers_df: pd.DataFrame, 
                                 transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate customer segments based on spending patterns"""
        print("Generating customer segments...")
        
        customer_metrics = transactions_df.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        customer_metrics.columns = [
            'customer_id', 'total_spent', 'avg_amount', 'transaction_count',
            'first_transaction', 'last_transaction'
        ]
        
        customer_metrics['customer_lifetime_days'] = (
            pd.to_datetime(customer_metrics['last_transaction']) - 
            pd.to_datetime(customer_metrics['first_transaction'])
        ).dt.days
        
        customer_metrics['avg_monthly_spend'] = customer_metrics['total_spent'] / (
            customer_metrics['customer_lifetime_days'] / 30
        ).clip(lower=1)
        
        segments = []
        for _, customer in customer_metrics.iterrows():
            total_spent = customer['total_spent']
            avg_monthly = customer['avg_monthly_spend']
            transaction_count = customer['transaction_count']
            
            if total_spent > 5000 and avg_monthly > 200:
                segment_name = "High-Value-Frequent-Active"
            elif total_spent > 2000 and transaction_count > 20:
                segment_name = "Medium-Value-Regular-Active"
            elif total_spent > 1000 and transaction_count > 10:
                segment_name = "Medium-Value-Occasional-Active"
            elif total_spent > 500:
                segment_name = "Low-Value-Occasional-Active"
            else:
                segment_name = "Low-Value-Rare-Inactive"
            
            segment = {
                'customer_id': customer['customer_id'],
                'segment_id': len(segments),
                'segment_name': segment_name,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            segments.append(segment)
        
        segments_df = pd.DataFrame(segments)
        print(f"Generated {len(segments_df)} customer segments")
        return segments_df
    
    def generate_campaigns(self, customers_df: pd.DataFrame, 
                          segments_df: pd.DataFrame) -> pd.DataFrame:
        """Generate marketing campaigns"""
        print("Generating marketing campaigns...")
        
        campaigns = []
        campaign_id = 1
        
        segment_campaigns = {
            "High-Value-Frequent-Active": {
                'types': ['loyalty_reward', 'exclusive_offer'],
                'discount_range': (15, 25)
            },
            "Medium-Value-Regular-Active": {
                'types': ['loyalty_reward', 'cross_sell'],
                'discount_range': (10, 20)
            },
            "Medium-Value-Occasional-Active": {
                'types': ['cross_sell', 'seasonal_promotion'],
                'discount_range': (15, 25)
            },
            "Low-Value-Occasional-Active": {
                'types': ['win_back', 'seasonal_promotion'],
                'discount_range': (20, 30)
            },
            "Low-Value-Rare-Inactive": {
                'types': ['win_back', 'reactivation'],
                'discount_range': (25, 35)
            }
        }
        
        for segment_name, campaign_config in segment_campaigns.items():
            segment_customers = segments_df[segments_df['segment_name'] == segment_name]
            
            for _, customer_segment in segment_customers.iterrows():
                num_campaigns = random.randint(1, 3)
                
                for _ in range(num_campaigns):
                    campaign_type = random.choice(campaign_config['types'])
                    discount = random.randint(*campaign_config['discount_range'])
                    
                    campaign = {
                        'campaign_id': f'CAMP_{campaign_id:06d}',
                        'customer_id': customer_segment['customer_id'],
                        'campaign_type': campaign_type,
                        'offer_description': f'{campaign_type.replace("_", " ").title()} for {discount}% off',
                        'discount_amount': discount,
                        'created_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                        'status': random.choice(['active', 'sent', 'expired'])
                    }
                    campaigns.append(campaign)
                    campaign_id += 1
        
        campaigns_df = pd.DataFrame(campaigns)
        print(f"Generated {len(campaigns_df)} campaigns")
        return campaigns_df
    
    def save_to_database(self, customers_df: pd.DataFrame, transactions_df: pd.DataFrame,
                        segments_df: pd.DataFrame, campaigns_df: pd.DataFrame):
        """Save all data to database"""
        print("Saving data to database...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Save customers
            customers_df.to_sql('customers', conn, if_exists='replace', index=False)
            
            # Save transactions
            transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
            
            # Save segments
            segments_df.to_sql('customer_segments', conn, if_exists='replace', index=False)
            
            # Save campaigns
            campaigns_df.to_sql('campaigns', conn, if_exists='replace', index=False)
        
        print("All data saved to database successfully!")
    
    def generate_all_data(self, num_customers: int = 1000):
        """Generate all sample data"""
        print("Starting comprehensive sample data generation...")
        
        # Generate customers
        customers_df = self.generate_customers(num_customers)
        
        # Generate transactions
        transactions_df = self.generate_transactions(customers_df)
        
        # Add seasonal patterns
        transactions_df = self.generate_seasonal_patterns(transactions_df)
        
        # Generate customer segments
        segments_df = self.generate_customer_segments(customers_df, transactions_df)
        
        # Generate campaigns
        campaigns_df = self.generate_campaigns(customers_df, segments_df)
        
        # Save to database
        self.save_to_database(customers_df, transactions_df, segments_df, campaigns_df)
        
        # Print summary
        self._print_data_summary(customers_df, transactions_df, segments_df, campaigns_df)
        
        print("Sample data generation completed successfully!")
    
    def _print_data_summary(self, customers_df: pd.DataFrame, transactions_df: pd.DataFrame,
                           segments_df: pd.DataFrame, campaigns_df: pd.DataFrame):
        """Print comprehensive data summary"""
        print("\n" + "="*60)
        print("📊 SAMPLE DATA GENERATION SUMMARY")
        print("="*60)
        
        print(f"👥 Customers: {len(customers_df):,}")
        print(f"🛒 Transactions: {len(transactions_df):,}")
        print(f"🏷️ Segments: {len(segments_df):,}")
        print(f"🎯 Campaigns: {len(campaigns_df):,}")
        
        print(f"\n💰 Total Revenue: ${transactions_df['amount'].sum():,.2f}")
        print(f"📈 Average Transaction: ${transactions_df['amount'].mean():.2f}")
        print(f"📅 Date Range: {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")
        
        print(f"\n🏪 Categories: {transactions_df['category'].nunique()}")
        print(f"🏪 Merchants: {transactions_df['merchant'].nunique()}")
        print(f"💳 Payment Methods: {transactions_df['payment_method'].nunique()}")
        
        print(f"\n📊 Segment Distribution:")
        segment_counts = segments_df['segment_name'].value_counts()
        for segment, count in segment_counts.items():
            print(f"   • {segment}: {count:,} customers")
        
        print(f"\n🎯 Campaign Distribution:")
        campaign_counts = campaigns_df['campaign_type'].value_counts()
        for campaign_type, count in campaign_counts.items():
            print(f"   • {campaign_type}: {count:,} campaigns")
        
        print("="*60)

def main():
    print("🚀 Starting Standalone Sample Data Generation...")
    
    try:
        generator = StandaloneDataGenerator()
        generator.generate_all_data(num_customers=1000)
        print("✅ Sample data generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during sample data generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 