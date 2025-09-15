"""
Customer Intelligence API Client for Real-time Analytics and Insights
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict
import logging

from src.utils.logger import setup_logger


@dataclass
class CustomerEvent:
    """Customer behavior event"""
    customer_id: str
    event_type: str
    timestamp: datetime
    properties: Dict[str, Any]
    session_id: Optional[str] = None
    source: str = 'api'


@dataclass
class CustomerInsight:
    """Customer insight data"""
    customer_id: str
    segment: str
    lifetime_value: float
    churn_probability: float
    next_best_action: str
    personalized_recommendations: List[str]
    risk_score: float
    engagement_score: float
    last_updated: datetime


class CustomerIntelligenceClient:
    """Client for Customer Intelligence API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.cicop.com/v2"):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = setup_logger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'CICOP-Python-Client/1.0'
        })
        
    def get_customer_insights(self, 
                             customer_id: str,
                             include_predictions: bool = True,
                             include_recommendations: bool = True) -> CustomerInsight:
        """Get comprehensive customer insights"""
        
        endpoint = f"{self.base_url}/customers/{customer_id}/insights"
        params = {
            'include_predictions': include_predictions,
            'include_recommendations': include_recommendations
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to CustomerInsight object
            insight = CustomerInsight(
                customer_id=data['customer_id'],
                segment=data['segment'],
                lifetime_value=data['lifetime_value'],
                churn_probability=data['churn_probability'],
                next_best_action=data['next_best_action'],
                personalized_recommendations=data['personalized_recommendations'],
                risk_score=data['risk_score'],
                engagement_score=data['engagement_score'],
                last_updated=datetime.fromisoformat(data['last_updated'])
            )
            
            return insight
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching customer insights: {e}")
            raise
    
    def get_customer_segments(self, 
                             filters: Optional[Dict[str, Any]] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get customer segments with optional filtering"""
        
        endpoint = f"{self.base_url}/segments"
        params = {'limit': limit}
        
        if filters:
            params.update(filters)
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            return response.json()['segments']
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching customer segments: {e}")
            raise
    
    def get_segment_analytics(self, 
                             segment_id: str,
                             metrics: List[str],
                             time_range: str = '30d') -> Dict[str, Any]:
        """Get analytics for a specific segment"""
        
        endpoint = f"{self.base_url}/segments/{segment_id}/analytics"
        params = {
            'metrics': ','.join(metrics),
            'time_range': time_range
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching segment analytics: {e}")
            raise
    
    def get_campaign_performance(self, 
                                campaign_ids: Optional[List[str]] = None,
                                time_range: str = '30d') -> Dict[str, Any]:
        """Get campaign performance metrics"""
        
        endpoint = f"{self.base_url}/campaigns/performance"
        params = {'time_range': time_range}
        
        if campaign_ids:
            params['campaign_ids'] = ','.join(campaign_ids)
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching campaign performance: {e}")
            raise
    
    def create_customer_event(self, event: CustomerEvent) -> bool:
        """Create a customer behavior event"""
        
        endpoint = f"{self.base_url}/events"
        event_data = asdict(event)
        event_data['timestamp'] = event.timestamp.isoformat()
        
        try:
            response = self.session.post(endpoint, json=event_data)
            response.raise_for_status()
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error creating customer event: {e}")
            return False
    
    def stream_customer_events(self, 
                             customer_id: Optional[str] = None,
                             event_types: Optional[List[str]] = None,
                             batch_size: int = 100) -> Generator[CustomerEvent, None, None]:
        """Stream customer events in real-time"""
        
        endpoint = f"{self.base_url}/events/stream"
        params = {'batch_size': batch_size}
        
        if customer_id:
            params['customer_id'] = customer_id
        if event_types:
            params['event_types'] = ','.join(event_types)
        
        try:
            with self.session.get(endpoint, params=params, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            event_data = json.loads(line.decode('utf-8'))
                            
                            # Convert to CustomerEvent object
                            event = CustomerEvent(
                                customer_id=event_data['customer_id'],
                                event_type=event_data['event_type'],
                                timestamp=datetime.fromisoformat(event_data['timestamp']),
                                properties=event_data['properties'],
                                session_id=event_data.get('session_id'),
                                source=event_data.get('source', 'api')
                            )
                            
                            yield event
                            
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse event data: {e}")
                            continue
                            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error streaming customer events: {e}")
            raise
    
    def get_realtime_metrics(self, 
                            metric_names: List[str],
                            dimensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get real-time metrics"""
        
        endpoint = f"{self.base_url}/metrics/realtime"
        params = {
            'metrics': ','.join(metric_names)
        }
        
        if dimensions:
            params['dimensions'] = ','.join(dimensions)
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching real-time metrics: {e}")
            raise
    
    def get_predictions(self, 
                       customer_ids: List[str],
                       prediction_types: List[str]) -> Dict[str, Any]:
        """Get ML predictions for customers"""
        
        endpoint = f"{self.base_url}/predictions"
        data = {
            'customer_ids': customer_ids,
            'prediction_types': prediction_types
        }
        
        try:
            response = self.session.post(endpoint, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching predictions: {e}")
            raise
    
    def get_recommendations(self, 
                           customer_id: str,
                           recommendation_types: List[str],
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a customer"""
        
        endpoint = f"{self.base_url}/customers/{customer_id}/recommendations"
        params = {
            'types': ','.join(recommendation_types),
            'limit': limit
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            return response.json()['recommendations']
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching recommendations: {e}")
            raise
    
    def create_alert(self, 
                     alert_type: str,
                     conditions: Dict[str, Any],
                     actions: List[Dict[str, Any]]) -> str:
        """Create a real-time alert"""
        
        endpoint = f"{self.base_url}/alerts"
        data = {
            'alert_type': alert_type,
            'conditions': conditions,
            'actions': actions,
            'status': 'active'
        }
        
        try:
            response = self.session.post(endpoint, json=data)
            response.raise_for_status()
            
            return response.json()['alert_id']
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error creating alert: {e}")
            raise
    
    def get_alerts(self, 
                   status: Optional[str] = None,
                   alert_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get configured alerts"""
        
        endpoint = f"{self.base_url}/alerts"
        params = {}
        
        if status:
            params['status'] = status
        if alert_types:
            params['types'] = ','.join(alert_types)
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            return response.json()['alerts']
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching alerts: {e}")
            raise
    
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert"""
        
        endpoint = f"{self.base_url}/alerts/{alert_id}"
        
        try:
            response = self.session.patch(endpoint, json=updates)
            response.raise_for_status()
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error updating alert: {e}")
            return False
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert"""
        
        endpoint = f"{self.base_url}/alerts/{alert_id}"
        
        try:
            response = self.session.delete(endpoint)
            response.raise_for_status()
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error deleting alert: {e}")
            return False
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and health information"""
        
        endpoint = f"{self.base_url}/health"
        
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching API status: {e}")
            raise
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        
        try:
            response = self.session.get(f"{self.base_url}/rate-limits")
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching rate limit info: {e}")
            raise
    
    def close(self):
        """Close the client session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for common operations
def create_customer_event_simple(customer_id: str, 
                                event_type: str, 
                                properties: Dict[str, Any],
                                api_key: str,
                                base_url: str = "https://api.cicop.com/v2") -> bool:
    """Simple function to create a customer event"""
    
    with CustomerIntelligenceClient(api_key, base_url) as client:
        event = CustomerEvent(
            customer_id=customer_id,
            event_type=event_type,
            timestamp=datetime.now(),
            properties=properties
        )
        return client.create_customer_event(event)


def get_customer_insights_simple(customer_id: str,
                                api_key: str,
                                base_url: str = "https://api.cicop.com/v2") -> Optional[CustomerInsight]:
    """Simple function to get customer insights"""
    
    try:
        with CustomerIntelligenceClient(api_key, base_url) as client:
            return client.get_customer_insights(customer_id)
    except Exception as e:
        logging.error(f"Error getting customer insights: {e}")
        return None
