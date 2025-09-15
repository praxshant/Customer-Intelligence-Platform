"""
Advanced Campaign Orchestrator for Multi-Channel Campaign Management
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import random
import json
from dataclasses import dataclass, asdict

from .campaign_generator import CampaignGenerator
from .rules_engine import RulesEngine
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


@dataclass
class CampaignChannel:
    """Campaign channel configuration"""
    name: str
    priority: int
    cost_per_message: float
    delivery_speed: str  # 'instant', 'fast', 'standard'
    personalization_level: str  # 'basic', 'advanced', 'dynamic'
    max_frequency: int  # messages per day


@dataclass
class ABTestConfig:
    """A/B testing configuration"""
    test_name: str
    variants: List[str]
    traffic_split: List[float]
    primary_metric: str
    secondary_metrics: List[str]
    test_duration_days: int
    minimum_sample_size: int


class CampaignOrchestrator:
    """Advanced campaign orchestrator for multi-channel campaign management"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(__name__)
        self.campaign_generator = CampaignGenerator()
        self.rules_engine = RulesEngine()
        
        # Initialize channels
        self.channels = self._initialize_channels()
        
        # A/B testing registry
        self.ab_tests = {}
        
        # Campaign performance tracking
        self.performance_metrics = {}
        
    def _initialize_channels(self) -> Dict[str, CampaignChannel]:
        """Initialize campaign channels with default configurations"""
        channels = {
            'email': CampaignChannel(
                name='email',
                priority=1,
                cost_per_message=0.01,
                delivery_speed='fast',
                personalization_level='advanced',
                max_frequency=2
            ),
            'sms': CampaignChannel(
                name='sms',
                priority=2,
                cost_per_message=0.05,
                delivery_speed='instant',
                personalization_level='basic',
                max_frequency=1
            ),
            'push': CampaignChannel(
                name='push',
                priority=3,
                cost_per_message=0.001,
                delivery_speed='instant',
                personalization_level='dynamic',
                max_frequency=3
            ),
            'social': CampaignChannel(
                name='social',
                priority=4,
                cost_per_message=0.02,
                delivery_speed='standard',
                personalization_level='advanced',
                max_frequency=1
            )
        }
        return channels
    
    def generate_multi_channel_campaigns(self, 
                                       target_segments: List[str],
                                       channels: List[str],
                                       personalization_level: str = 'advanced',
                                       a_b_testing: bool = False,
                                       budget_constraint: float = 10000) -> Dict[str, Any]:
        """Generate multi-channel campaigns with advanced orchestration"""
        
        self.logger.info(f"Generating multi-channel campaigns for segments: {target_segments}")
        
        # Step 1: Generate base campaigns for each segment
        all_campaigns = {}
        for segment in target_segments:
            segment_campaigns = self.campaign_generator.generate_segment_campaigns(
                pd.DataFrame({'segment_name': [segment]}), segment
            )
            all_campaigns[segment] = segment_campaigns
        
        # Step 2: Create multi-channel variants
        multi_channel_campaigns = {}
        for segment, campaigns in all_campaigns.items():
            if isinstance(campaigns, list):
                campaigns = pd.DataFrame(campaigns)
            
            segment_multi_channel = []
            for _, campaign in campaigns.iterrows():
                channel_variants = self._create_channel_variants(
                    campaign, channels, personalization_level
                )
                segment_multi_channel.extend(channel_variants)
            
            multi_channel_campaigns[segment] = segment_multi_channel
        
        # Step 3: Apply A/B testing if requested
        if a_b_testing:
            multi_channel_campaigns = self._apply_ab_testing(multi_channel_campaigns)
        
        # Step 4: Optimize across channels
        optimized_campaigns = self._optimize_multi_channel_portfolio(
            multi_channel_campaigns, budget_constraint
        )
        
        # Step 5: Schedule campaigns
        scheduled_campaigns = self._schedule_multi_channel_campaigns(optimized_campaigns)
        
        return {
            'campaigns': scheduled_campaigns,
            'channel_distribution': self._get_channel_distribution(scheduled_campaigns),
            'budget_allocation': self._calculate_budget_allocation(scheduled_campaigns),
            'expected_roi': self._estimate_roi(scheduled_campaigns),
            'ab_tests': list(self.ab_tests.keys()) if a_b_testing else []
        }
    
    def _create_channel_variants(self, campaign: pd.Series, 
                                channels: List[str], 
                                personalization_level: str) -> List[Dict[str, Any]]:
        """Create channel-specific variants of a campaign"""
        variants = []
        
        for channel in channels:
            if channel not in self.channels:
                continue
            
            channel_config = self.channels[channel]
            
            # Create channel-specific campaign
            channel_campaign = campaign.to_dict()
            channel_campaign['campaign_id'] = f"{campaign['campaign_id']}_{channel}"
            channel_campaign['channel'] = channel
            channel_campaign['channel_priority'] = channel_config.priority
            channel_campaign['delivery_speed'] = channel_config.delivery_speed
            channel_campaign['personalization_level'] = personalization_level
            
            # Adjust campaign parameters based on channel
            channel_campaign = self._adapt_campaign_for_channel(
                channel_campaign, channel_config, personalization_level
            )
            
            variants.append(channel_campaign)
        
        return variants
    
    def _adapt_campaign_for_channel(self, campaign: Dict[str, Any], 
                                   channel_config: CampaignChannel,
                                   personalization_level: str) -> Dict[str, Any]:
        """Adapt campaign parameters for specific channel"""
        
        # Adjust discount based on channel cost
        base_discount = campaign.get('discount_percentage', 10)
        if channel_config.name == 'sms':
            # SMS campaigns need higher discounts due to cost
            campaign['discount_percentage'] = min(base_discount * 1.2, 30)
        elif channel_config.name == 'push':
            # Push notifications can have lower discounts
            campaign['discount_percentage'] = max(base_discount * 0.8, 5)
        
        # Adjust personalization based on channel capability
        if personalization_level == 'dynamic' and channel_config.personalization_level == 'dynamic':
            campaign['dynamic_content'] = True
            campaign['personalization_fields'] = ['name', 'location', 'preferences', 'behavior']
        elif personalization_level == 'advanced':
            campaign['personalization_fields'] = ['name', 'preferences']
        else:
            campaign['personalization_fields'] = ['name']
        
        # Add channel-specific metadata
        campaign['channel_cost'] = channel_config.cost_per_message
        campaign['max_frequency'] = channel_config.max_frequency
        campaign['estimated_delivery_time'] = self._estimate_delivery_time(channel_config)
        
        return campaign
    
    def _estimate_delivery_time(self, channel_config: CampaignChannel) -> str:
        """Estimate delivery time for a channel"""
        if channel_config.delivery_speed == 'instant':
            return '0-5 minutes'
        elif channel_config.delivery_speed == 'fast':
            return '5-30 minutes'
        else:
            return '1-4 hours'
    
    def _apply_ab_testing(self, campaigns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Apply A/B testing to campaigns"""
        
        for segment, segment_campaigns in campaigns.items():
            if not segment_campaigns:
                continue
            
            # Create A/B test for this segment
            test_config = ABTestConfig(
                test_name=f"segment_{segment}_test_{datetime.now().strftime('%Y%m%d')}",
                variants=['control', 'variant_a', 'variant_b'],
                traffic_split=[0.33, 0.33, 0.34],
                primary_metric='conversion_rate',
                secondary_metrics=['click_through_rate', 'revenue_per_customer'],
                test_duration_days=14,
                minimum_sample_size=100
            )
            
            self.ab_tests[test_config.test_name] = test_config
            
            # Assign variants to campaigns
            for i, campaign in enumerate(segment_campaigns):
                variant = test_config.variants[i % len(test_config.variants)]
                campaign['ab_test_variant'] = variant
                campaign['ab_test_name'] = test_config.test_name
                campaign['traffic_split'] = test_config.traffic_split[i % len(test_config.variants)]
        
        return campaigns
    
    def _optimize_multi_channel_portfolio(self, campaigns: Dict[str, List[Dict[str, Any]]], 
                                        budget_constraint: float) -> Dict[str, List[Dict[str, Any]]]:
        """Optimize campaign portfolio across channels"""
        
        # Flatten all campaigns
        all_campaigns = []
        for segment_campaigns in campaigns.values():
            all_campaigns.extend(segment_campaigns)
        
        if not all_campaigns:
            return campaigns
        
        # Calculate campaign scores
        for campaign in all_campaigns:
            campaign['portfolio_score'] = self._calculate_portfolio_score(campaign)
        
        # Sort by portfolio score
        all_campaigns.sort(key=lambda x: x['portfolio_score'], reverse=True)
        
        # Select campaigns within budget
        selected_campaigns = []
        total_cost = 0
        
        for campaign in all_campaigns:
            campaign_cost = self._calculate_campaign_cost(campaign)
            if total_cost + campaign_cost <= budget_constraint:
                selected_campaigns.append(campaign)
                total_cost += campaign_cost
            else:
                break
        
        # Reorganize by segment
        optimized_campaigns = {}
        for campaign in selected_campaigns:
            segment = campaign.get('target_segment', 'Unknown')
            if segment not in optimized_campaigns:
                optimized_campaigns[segment] = []
            optimized_campaigns[segment].append(campaign)
        
        self.logger.info(f"Portfolio optimization complete: {len(selected_campaigns)} campaigns selected, "
                        f"budget used: ${total_cost:.2f}")
        
        return optimized_campaigns
    
    def _calculate_portfolio_score(self, campaign: Dict[str, Any]) -> float:
        """Calculate portfolio score for campaign selection"""
        score = 0.0
        
        # Base score from campaign priority
        score += campaign.get('priority_score', 0.5) * 0.3
        
        # Channel efficiency score
        channel = campaign.get('channel', 'email')
        if channel in self.channels:
            channel_config = self.channels[channel]
            score += (1.0 / channel_config.cost_per_message) * 0.2
        
        # Personalization score
        personalization_level = campaign.get('personalization_level', 'basic')
        personalization_scores = {'basic': 0.5, 'advanced': 0.8, 'dynamic': 1.0}
        score += personalization_scores.get(personalization_level, 0.5) * 0.2
        
        # Segment value score
        segment = campaign.get('target_segment', 'Unknown')
        segment_values = {'High Value': 1.0, 'Medium Value': 0.7, 'Low Value': 0.4}
        score += segment_values.get(segment, 0.5) * 0.3
        
        return min(score, 1.0)
    
    def _calculate_campaign_cost(self, campaign: Dict[str, Any]) -> float:
        """Calculate total cost for a campaign"""
        base_cost = campaign.get('channel_cost', 0.01)
        
        # Adjust cost based on personalization level
        personalization_level = campaign.get('personalization_level', 'basic')
        personalization_multipliers = {'basic': 1.0, 'advanced': 1.2, 'dynamic': 1.5}
        multiplier = personalization_multipliers.get(personalization_level, 1.0)
        
        return base_cost * multiplier
    
    def _schedule_multi_channel_campaigns(self, campaigns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Schedule campaigns across channels with timing optimization"""
        
        scheduled_campaigns = {}
        
        for segment, segment_campaigns in campaigns.items():
            scheduled_segment = []
            
            for campaign in segment_campaigns:
                # Add scheduling information
                campaign['scheduled_time'] = self._calculate_optimal_timing(campaign)
                campaign['delivery_window'] = self._calculate_delivery_window(campaign)
                campaign['retry_schedule'] = self._generate_retry_schedule(campaign)
                
                scheduled_segment.append(campaign)
            
            scheduled_campaigns[segment] = scheduled_segment
        
        return scheduled_campaigns
    
    def _calculate_optimal_timing(self, campaign: Dict[str, Any]) -> str:
        """Calculate optimal timing for campaign delivery"""
        # Simple timing logic - can be enhanced with ML-based timing optimization
        channel = campaign.get('channel', 'email')
        
        if channel == 'email':
            return '09:00'  # Best time for email engagement
        elif channel == 'sms':
            return '12:00'  # Lunch time for SMS
        elif channel == 'push':
            return '18:00'  # Evening for push notifications
        else:
            return '10:00'  # Default morning time
    
    def _calculate_delivery_window(self, campaign: Dict[str, Any]) -> str:
        """Calculate delivery window for campaign"""
        channel = campaign.get('channel', 'email')
        
        if channel == 'sms':
            return '2 hours'  # SMS should be delivered quickly
        elif channel == 'push':
            return '1 hour'   # Push notifications are time-sensitive
        else:
            return '4 hours'  # Email can have longer delivery window
    
    def _generate_retry_schedule(self, campaign: Dict[str, Any]) -> List[str]:
        """Generate retry schedule for failed deliveries"""
        channel = campaign.get('channel', 'email')
        
        if channel == 'sms':
            return ['1 hour', '4 hours', '24 hours']  # SMS retries
        elif channel == 'push':
            return ['30 minutes', '2 hours', '6 hours']  # Push retries
        else:
            return ['2 hours', '6 hours', '24 hours']  # Email retries
    
    def _get_channel_distribution(self, campaigns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """Get distribution of campaigns across channels"""
        channel_counts = {}
        
        for segment_campaigns in campaigns.values():
            for campaign in segment_campaigns:
                channel = campaign.get('channel', 'unknown')
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        return channel_counts
    
    def _calculate_budget_allocation(self, campaigns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate budget allocation across channels"""
        channel_budgets = {}
        
        for segment_campaigns in campaigns.values():
            for campaign in segment_campaigns:
                channel = campaign.get('channel', 'unknown')
                cost = self._calculate_campaign_cost(campaign)
                channel_budgets[channel] = channel_budgets.get(channel, 0) + cost
        
        return channel_budgets
    
    def _estimate_roi(self, campaigns: Dict[str, List[Dict[str, Any]]]) -> float:
        """Estimate ROI for the campaign portfolio"""
        total_cost = 0
        total_expected_revenue = 0
        
        for segment_campaigns in campaigns.values():
            for campaign in segment_campaigns:
                cost = self._calculate_campaign_cost(campaign)
                total_cost += cost
                
                # Estimate revenue based on campaign type and segment
                base_revenue = campaign.get('min_purchase', 50)
                segment = campaign.get('target_segment', 'Medium Value')
                segment_multipliers = {'High Value': 2.0, 'Medium Value': 1.5, 'Low Value': 1.0}
                multiplier = segment_multipliers.get(segment, 1.0)
                
                expected_revenue = base_revenue * multiplier * 0.1  # 10% conversion rate
                total_expected_revenue += expected_revenue
        
        if total_cost > 0:
            return (total_expected_revenue - total_cost) / total_cost
        else:
            return 0.0
    
    def get_campaign_analytics(self, campaign_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get analytics for campaigns"""
        # This would integrate with actual analytics data
        # For now, return mock analytics
        return {
            'total_campaigns': len(self.ab_tests) * 3,  # 3 variants per test
            'active_tests': len(self.ab_tests),
            'channel_performance': {
                'email': {'open_rate': 0.25, 'click_rate': 0.03, 'conversion_rate': 0.01},
                'sms': {'delivery_rate': 0.98, 'response_rate': 0.05, 'conversion_rate': 0.02},
                'push': {'delivery_rate': 0.95, 'open_rate': 0.40, 'conversion_rate': 0.015}
            },
            'ab_test_results': {
                test_name: {
                    'status': 'running',
                    'days_remaining': 7,
                    'current_sample_size': 150
                }
                for test_name in self.ab_tests.keys()
            }
        }
    
    def pause_campaign(self, campaign_id: str) -> bool:
        """Pause a specific campaign"""
        # Implementation would update campaign status in database
        self.logger.info(f"Campaign {campaign_id} paused")
        return True
    
    def resume_campaign(self, campaign_id: str) -> bool:
        """Resume a paused campaign"""
        # Implementation would update campaign status in database
        self.logger.info(f"Campaign {campaign_id} resumed")
        return True
    
    def update_campaign_budget(self, campaign_id: str, new_budget: float) -> bool:
        """Update budget for a specific campaign"""
        # Implementation would update campaign budget in database
        self.logger.info(f"Campaign {campaign_id} budget updated to ${new_budget}")
        return True
