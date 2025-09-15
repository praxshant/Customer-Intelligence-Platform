# test_rules_engine.py
# Purpose: Unit tests for rules engine functionality

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rules.rules_engine import RulesEngine
from rules.campaign_generator import CampaignGenerator
from utils.logger import setup_logger

class TestRulesEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and rules engine instance"""
        self.rules_engine = RulesEngine()
        self.campaign_generator = CampaignGenerator()
        self.logger = setup_logger("TestRulesEngine")
        
        # Create sample customer data
        self.sample_customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(50)],
            'first_name': [f'Customer{i}' for i in range(50)],
            'last_name': [f'Test{i}' for i in range(50)],
            'email': [f'customer{i}@test.com' for i in range(50)],
            'registration_date': pd.date_range('2020-01-01', periods=50),
            'age': np.random.randint(18, 80, 50),
            'gender': np.random.choice(['M', 'F'], 50),
            'location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 50),
            'total_spent': np.random.uniform(100, 5000, 50),
            'transaction_count': np.random.randint(5, 100, 50),
            'avg_transaction_amount': np.random.uniform(20, 200, 50),
            'recency_days': np.random.randint(1, 365, 50),
            'segment_name': np.random.choice(['High Value', 'Medium Value', 'Low Value'], 50)
        })
        
        # Create sample transaction data
        self.sample_transactions = pd.DataFrame({
            'customer_id': np.random.choice(self.sample_customers['customer_id'], 200),
            'transaction_date': pd.date_range('2023-01-01', periods=200),
            'amount': np.random.uniform(10, 500, 200),
            'category': np.random.choice(['Food', 'Shopping', 'Entertainment', 'Transport'], 200),
            'merchant': [f'Merchant_{i}' for i in range(200)]
        })
    
    def test_campaign_eligibility_check(self):
        """Test customer eligibility for different campaign types"""
        # Test loyalty reward eligibility
        customer = self.sample_customers.iloc[0]
        customer_dict = customer.to_dict()
        
        # Test high-value customer for loyalty reward
        customer_dict['total_spent'] = 5000
        customer_dict['transaction_count'] = 50
        customer_dict['recency_days'] = 30
        
        eligibility = self.rules_engine.check_campaign_eligibility(
            customer_dict, 'loyalty_reward'
        )
        self.assertTrue(eligibility['eligible'])
        self.assertIn('discount_amount', eligibility)
        self.assertIn('offer_description', eligibility)
        
        # Test win-back campaign eligibility
        customer_dict['recency_days'] = 200
        eligibility = self.rules_engine.check_campaign_eligibility(
            customer_dict, 'win_back'
        )
        self.assertTrue(eligibility['eligible'])
        
        # Test cross-sell eligibility
        eligibility = self.rules_engine.check_campaign_eligibility(
            customer_dict, 'cross_sell'
        )
        self.assertIsInstance(eligibility['eligible'], bool)
    
    def test_campaign_creation(self):
        """Test campaign creation functionality"""
        customer = self.sample_customers.iloc[0]
        customer_dict = customer.to_dict()
        
        # Create a loyalty campaign
        campaign = self.rules_engine.create_campaign(
            customer_dict, 'loyalty_reward'
        )
        
        # Check campaign structure
        self.assertIsInstance(campaign, dict)
        self.assertIn('campaign_id', campaign)
        self.assertIn('customer_id', campaign)
        self.assertIn('campaign_type', campaign)
        self.assertIn('offer_description', campaign)
        self.assertIn('discount_amount', campaign)
        self.assertIn('created_date', campaign)
        self.assertIn('status', campaign)
        
        # Check campaign values
        self.assertEqual(campaign['customer_id'], customer_dict['customer_id'])
        self.assertEqual(campaign['campaign_type'], 'loyalty_reward')
        self.assertEqual(campaign['status'], 'active')
        self.assertGreater(campaign['discount_amount'], 0)
    
    def test_campaign_priority_scoring(self):
        """Test campaign priority scoring"""
        customer = self.sample_customers.iloc[0]
        customer_dict = customer.to_dict()
        
        # Test priority scoring for different campaign types
        loyalty_score = self.rules_engine.calculate_campaign_priority(
            customer_dict, 'loyalty_reward'
        )
        self.assertIsInstance(loyalty_score, float)
        self.assertGreaterEqual(loyalty_score, 0)
        self.assertLessEqual(loyalty_score, 100)
        
        win_back_score = self.rules_engine.calculate_campaign_priority(
            customer_dict, 'win_back'
        )
        self.assertIsInstance(win_back_score, float)
    
    def test_bulk_campaign_generation(self):
        """Test bulk campaign generation"""
        # Generate campaigns for all customers
        campaigns = self.rules_engine.generate_bulk_campaigns(
            self.sample_customers, campaign_type='loyalty_reward'
        )
        
        # Check output
        self.assertIsInstance(campaigns, list)
        self.assertGreater(len(campaigns), 0)
        
        # Check campaign structure
        for campaign in campaigns:
            self.assertIn('campaign_id', campaign)
            self.assertIn('customer_id', campaign)
            self.assertIn('campaign_type', campaign)
            self.assertEqual(campaign['campaign_type'], 'loyalty_reward')
    
    def test_seasonal_campaign_adjustments(self):
        """Test seasonal campaign adjustments"""
        customer = self.sample_customers.iloc[0]
        customer_dict = customer.to_dict()
        
        # Test winter seasonal adjustment
        winter_campaign = self.rules_engine.apply_seasonal_adjustments(
            customer_dict, 'seasonal_promotion', 'winter'
        )
        self.assertIn('seasonal_multiplier', winter_campaign)
        self.assertIn('adjusted_discount', winter_campaign)
        
        # Test summer seasonal adjustment
        summer_campaign = self.rules_engine.apply_seasonal_adjustments(
            customer_dict, 'seasonal_promotion', 'summer'
        )
        self.assertNotEqual(winter_campaign['adjusted_discount'], 
                           summer_campaign['adjusted_discount'])
    
    def test_budget_based_filtering(self):
        """Test budget-based campaign filtering"""
        # Generate campaigns
        campaigns = self.rules_engine.generate_bulk_campaigns(
            self.sample_customers, campaign_type='loyalty_reward'
        )
        
        # Test budget filtering
        total_budget = 10000
        filtered_campaigns = self.rules_engine.filter_campaigns_by_budget(
            campaigns, total_budget
        )
        
        # Check that total discount amount doesn't exceed budget
        total_discount = sum(c['discount_amount'] for c in filtered_campaigns)
        self.assertLessEqual(total_discount, total_budget)
    
    def test_campaign_validation(self):
        """Test campaign validation rules"""
        # Test valid campaign
        valid_campaign = {
            'campaign_id': 'CAMP_001',
            'customer_id': 'CUST_001',
            'campaign_type': 'loyalty_reward',
            'discount_amount': 50.0,
            'created_date': datetime.now(),
            'status': 'active'
        }
        
        is_valid = self.rules_engine.validate_campaign(valid_campaign)
        self.assertTrue(is_valid)
        
        # Test invalid campaign (negative discount)
        invalid_campaign = valid_campaign.copy()
        invalid_campaign['discount_amount'] = -10.0
        
        is_valid = self.rules_engine.validate_campaign(invalid_campaign)
        self.assertFalse(is_valid)

class TestCampaignGenerator(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and campaign generator instance"""
        self.campaign_generator = CampaignGenerator()
        self.rules_engine = RulesEngine()
        
        # Create sample data
        self.sample_customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(20)],
            'segment_name': ['High Value'] * 5 + ['Medium Value'] * 10 + ['Low Value'] * 5,
            'total_spent': np.random.uniform(100, 5000, 20),
            'transaction_count': np.random.randint(5, 100, 20),
            'recency_days': np.random.randint(1, 365, 20)
        })
    
    def test_segment_specific_campaigns(self):
        """Test segment-specific campaign generation"""
        # Generate campaigns for high-value segment
        high_value_customers = self.sample_customers[
            self.sample_customers['segment_name'] == 'High Value'
        ]
        
        campaigns = self.campaign_generator.generate_segment_campaigns(
            high_value_customers, 'High Value'
        )
        
        # Check output
        self.assertIsInstance(campaigns, list)
        self.assertEqual(len(campaigns), len(high_value_customers))
        
                # Check campaign types for high-value customers
        for campaign in campaigns:
            self.assertIn(campaign['campaign_type'],
                         ['loyalty_reward', 'premium_offer', 'exclusive_deal', 'high_value_reward', 'seasonal_promotion'])
    
    def test_personalized_campaigns(self):
        """Test personalized campaign generation"""
        customer = self.sample_customers.iloc[0]
        
        campaigns = self.campaign_generator.generate_personalized_campaigns(
            customer.to_dict()
        )
        
        # Check output
        self.assertIsInstance(campaigns, list)
        self.assertGreater(len(campaigns), 0)
        
        # Check that campaigns are personalized
        for campaign in campaigns:
            self.assertIn('personalization_reason', campaign)
    
    def test_campaign_portfolio_optimization(self):
        """Test campaign portfolio optimization"""
        # Generate initial campaigns
        initial_campaigns = self.campaign_generator.generate_segment_campaigns(
            self.sample_customers, 'Medium Value'
        )
        
        # Optimize portfolio
        optimized_campaigns = self.campaign_generator.optimize_campaign_portfolio(
            initial_campaigns, max_budget=5000
        )
        
        # Check optimization results
        self.assertIsInstance(optimized_campaigns, dict)
        self.assertIn('selected_campaigns', optimized_campaigns)
        self.assertIn('total_budget_used', optimized_campaigns)
        self.assertIn('expected_roi', optimized_campaigns)
        
        # Check budget constraint
        self.assertLessEqual(optimized_campaigns['total_budget_used'], 5000)
    
    def test_campaign_reporting(self):
        """Test campaign reporting functionality"""
        # Generate sample campaigns
        campaigns = self.campaign_generator.generate_segment_campaigns(
            self.sample_customers, 'Low Value'
        )
        
        # Generate report
        report = self.campaign_generator.generate_campaign_report(campaigns)
        
        # Check report structure
        self.assertIsInstance(report, dict)
        self.assertIn('total_campaigns', report)
        self.assertIn('campaign_types', report)
        self.assertIn('total_budget', report)
        self.assertIn('segment_distribution', report)
        
        # Check report values
        self.assertEqual(report['total_campaigns'], len(campaigns))
        self.assertGreater(report['total_budget'], 0)
    
    def test_campaign_effectiveness_prediction(self):
        """Test campaign effectiveness prediction"""
        customer = self.sample_customers.iloc[0]
        customer_dict = customer.to_dict()
        
        # Predict effectiveness for different campaign types
        effectiveness = self.campaign_generator.predict_campaign_effectiveness(
            customer_dict, 'loyalty_reward'
        )
        
        # Check prediction structure
        self.assertIsInstance(effectiveness, dict)
        self.assertIn('response_probability', effectiveness)
        self.assertIn('expected_revenue', effectiveness)
        self.assertIn('roi_prediction', effectiveness)
        
        # Check prediction values
        self.assertGreaterEqual(effectiveness['response_probability'], 0)
        self.assertLessEqual(effectiveness['response_probability'], 1)
        self.assertGreaterEqual(effectiveness['expected_revenue'], 0)
    
    def test_campaign_scheduling(self):
        """Test campaign scheduling functionality"""
        campaigns = self.campaign_generator.generate_segment_campaigns(
            self.sample_customers, 'Medium Value'
        )
        
        # Schedule campaigns
        scheduled_campaigns = self.campaign_generator.schedule_campaigns(
            campaigns, start_date=datetime.now()
        )
        
        # Check scheduling
        self.assertIsInstance(scheduled_campaigns, list)
        for campaign in scheduled_campaigns:
            self.assertIn('scheduled_date', campaign)
            self.assertIn('delivery_channel', campaign)

if __name__ == '__main__':
    unittest.main() 