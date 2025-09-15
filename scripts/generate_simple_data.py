#!/usr/bin/env python3
# generate_simple_data.py
# Purpose: Generate comprehensive sample data without external dependencies

import sys
import os

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3

# Add project root to path so imports like `from src...` work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.database_manager import DatabaseManager
from src.utils.logger import setup_logger

class SimpleDataGenerator:
    def __init__(self):
        # Step 1: Initialize generator
        self.logger = setup_logger("SimpleDataGenerator")
        self.db_manager = DatabaseManager()
        
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
        
    def generate_customers(self, num_customers: int = 1000) -> pd.DataFrame:
        # Step 1: Generate customer data
        self.logger.info(f"Generating {num_customers} customers...")
        
        customers = []
        for i in range(num_customers):
            # Step 2: Generate realistic customer data
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
        self.logger.info(f"Generated {len(customers_df)} customers")
        return customers_df
    
    def generate_transactions(self, customers_df: pd.DataFrame, 
                            avg_transactions_per_customer: int = 15) -> pd.DataFrame:
        # Step 1: Generate transaction data
        total_transactions = len(customers_df) * avg_transactions_per_customer
        self.logger.info(f"Generating {total_transactions} transactions...")
        
        transactions = []
        transaction_id = 1
        
        for _, customer in customers_df.iterrows():
            # Step 2: Generate transactions for each customer
            num_transactions = random.randint(1, avg_transactions_per_customer * 2)
            
            # Step 3: Create transaction timeline
            start_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
            end_date = datetime.now()
            
            for _ in range(num_transactions):
                # Step 4: Generate realistic transaction
                days_between = (end_date - start_date).days
                random_days = random.randint(0, days_between)
                transaction_date = (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
                
                # Step 5: Select category and merchant
                category = random.choice(list(self.categories.keys()))
                merchant = random.choice(self.categories[category])
                
                # Step 6: Generate realistic amount based on category
                amount = self._generate_realistic_amount(category)
                
                # Step 7: Create transaction record
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
        
        # Step 8: Sort by date and remove duplicates
        transactions_df = transactions_df.sort_values('transaction_date')
        transactions_df = transactions_df.drop_duplicates(subset=['transaction_id'])
        
        self.logger.info(f"Generated {len(transactions_df)} transactions")
        return transactions_df
    
    def _generate_realistic_amount(self, category: str) -> float:
        # Step 1: Generate realistic amounts based on category
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
        
        # Step 2: Use log-normal distribution for realistic amounts
        mu = np.log((min_amount + max_amount) / 2)
        sigma = 0.5
        
        amount = np.random.lognormal(mu, sigma)
        amount = max(min_amount, min(max_amount, amount))
        
        return round(amount, 2)
    
    def _generate_subcategory(self, category: str) -> str:
        # Step 1: Generate realistic subcategories
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
        # Step 1: Add seasonal patterns to transactions
        self.logger.info("Adding seasonal patterns to transactions...")
        
        # Step 2: Create seasonal multipliers
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
        
        # Step 3: Apply seasonal adjustments
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        transactions_df['month'] = transactions_df['transaction_date'].dt.month
        
        # Step 4: Adjust amounts based on season
        for month, multiplier in seasonal_multipliers.items():
            mask = transactions_df['month'] == month
            transactions_df.loc[mask, 'amount'] = transactions_df.loc[mask, 'amount'] * multiplier
        
        # Step 5: Round amounts
        transactions_df['amount'] = transactions_df['amount'].round(2)
        
        # Step 6: Remove month column
        transactions_df = transactions_df.drop('month', axis=1)
        
        self.logger.info("Seasonal patterns applied successfully")
        return transactions_df
    
    def generate_customer_segments(self, customers_df: pd.DataFrame, 
                                 transactions_df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Generate customer segments based on spending patterns
        self.logger.info("Generating customer segments...")
        
        # Step 2: Calculate customer spending metrics
        customer_metrics = transactions_df.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        # Step 3: Flatten column names
        customer_metrics.columns = [
            'customer_id', 'total_spent', 'avg_amount', 'transaction_count',
            'first_transaction', 'last_transaction'
        ]
        
        # Step 4: Calculate additional metrics
        customer_metrics['customer_lifetime_days'] = (
            pd.to_datetime(customer_metrics['last_transaction']) - 
            pd.to_datetime(customer_metrics['first_transaction'])
        ).dt.days
        
        customer_metrics['avg_monthly_spend'] = customer_metrics['total_spent'] / (
            customer_metrics['customer_lifetime_days'] / 30
        ).clip(lower=1)
        
        # Step 5: Create segments based on spending behavior
        segments = []
        for _, customer in customer_metrics.iterrows():
            total_spent = customer['total_spent']
            avg_monthly = customer['avg_monthly_spend']
            transaction_count = customer['transaction_count']
            
            # Step 6: Determine segment based on behavior
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
        self.logger.info(f"Generated {len(segments_df)} customer segments")
        return segments_df
    
    def generate_campaigns(self, customers_df: pd.DataFrame, 
                          segments_df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Generate marketing campaigns
        self.logger.info("Generating marketing campaigns...")
        
        campaigns = []
        campaign_id = 1
        
        # Step 2: Generate campaigns for different segments
        segment_campaigns = {
            "High-Value-Frequent-Active": {
                'types': ['loyalty_reward', 'exclusive_offer'],
                'discount_range': (15, 25),
                'min_purchase': (100, 200)
            },
            "Medium-Value-Regular-Active": {
                'types': ['loyalty_reward', 'cross_sell'],
                'discount_range': (10, 20),
                'min_purchase': (75, 150)
            },
            "Medium-Value-Occasional-Active": {
                'types': ['cross_sell', 'seasonal_promotion'],
                'discount_range': (15, 25),
                'min_purchase': (50, 100)
            },
            "Low-Value-Occasional-Active": {
                'types': ['win_back', 'seasonal_promotion'],
                'discount_range': (20, 30),
                'min_purchase': (25, 75)
            },
            "Low-Value-Rare-Inactive": {
                'types': ['win_back', 'reactivation'],
                'discount_range': (25, 35),
                'min_purchase': (20, 50)
            }
        }
        
        # Step 3: Generate campaigns for each segment
        for segment_name, campaign_config in segment_campaigns.items():
            segment_customers = segments_df[segments_df['segment_name'] == segment_name]
            
            for _, customer_segment in segment_customers.iterrows():
                # Step 4: Generate 1-3 campaigns per customer
                num_campaigns = random.randint(1, 3)
                
                for _ in range(num_campaigns):
                    campaign_type = random.choice(campaign_config['types'])
                    discount = random.randint(*campaign_config['discount_range'])
                    min_purchase = random.randint(*campaign_config['min_purchase'])
                    
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
        self.logger.info(f"Generated {len(campaigns_df)} campaigns")
        return campaigns_df
    
    def save_to_database(self, customers_df: pd.DataFrame, transactions_df: pd.DataFrame,
                        segments_df: pd.DataFrame, campaigns_df: pd.DataFrame):
        # Step 1: Save all data to database
        self.logger.info("Saving data to database...")
        
        try:
            # Step 2: Save customers
            self.db_manager.insert_dataframe(customers_df, 'customers', if_exists='replace')
            
            # Step 3: Save transactions
            self.db_manager.insert_dataframe(transactions_df, 'transactions', if_exists='replace')
            
            # Step 4: Save segments
            self.db_manager.insert_dataframe(segments_df, 'customer_segments', if_exists='replace')
            
            # Step 5: Save campaigns
            self.db_manager.insert_dataframe(campaigns_df, 'campaigns', if_exists='replace')
            
            self.logger.info("All data saved to database successfully!")
            
        except Exception as e:
            self.logger.error(f"Error saving data to database: {str(e)}")
            raise
    
    def generate_all_data(self, num_customers: int = 1000):
        # Step 1: Generate all sample data
        self.logger.info("Starting comprehensive sample data generation...")
        
        # Step 2: Generate customers
        customers_df = self.generate_customers(num_customers)
        
        # Step 3: Generate transactions
        transactions_df = self.generate_transactions(customers_df)
        
        # Step 4: Add seasonal patterns
        transactions_df = self.generate_seasonal_patterns(transactions_df)
        
        # Step 5: Generate customer segments
        segments_df = self.generate_customer_segments(customers_df, transactions_df)
        
        # Step 6: Generate campaigns
        campaigns_df = self.generate_campaigns(customers_df, segments_df)
        
        # Step 7: Save to database
        self.save_to_database(customers_df, transactions_df, segments_df, campaigns_df)
        
        # Step 8: Print summary
        self._print_data_summary(customers_df, transactions_df, segments_df, campaigns_df)
        
        self.logger.info("Sample data generation completed successfully!")
    
    def _print_data_summary(self, customers_df: pd.DataFrame, transactions_df: pd.DataFrame,
                           segments_df: pd.DataFrame, campaigns_df: pd.DataFrame):
        # Step 1: Print comprehensive data summary
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
    # Step 1: Main execution
    print("🚀 Starting Simple Sample Data Generation...")
    
    try:
        # Step 2: Initialize generator
        generator = SimpleDataGenerator()
        
        # Step 3: Generate data (adjust number of customers as needed)
        generator.generate_all_data(num_customers=1000)
        
        print("✅ Sample data generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during sample data generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 