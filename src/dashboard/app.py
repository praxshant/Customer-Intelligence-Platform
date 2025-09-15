# app.py
# Purpose: Main Streamlit dashboard application

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys
import os
import io
import requests

# Add project root (directory containing 'src') to sys.path so 'import src.*' works
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.database_manager import DatabaseManager
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.ml.customer_segmentation import CustomerSegmentation
from src.ml.recommendation_engine import RecommendationEngine
from src.rules.campaign_generator import CampaignGenerator
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import calculate_date_range, format_currency, generate_color_palette

# Import dashboard components (use absolute imports to work when run as a script)
from src.dashboard.components.kpi_cards import render_kpi_cards
from src.dashboard.components.charts import render_spending_chart, render_segment_chart, render_category_chart
from src.dashboard.components.filters import render_filters
from src.dashboard.components.sidebar import render_sidebar

class DashboardApp:
    def __init__(self):
        # Step 1: Initialize dashboard components
        self.logger = setup_logger("DashboardApp")
        self.config = load_config()
        self.db_manager = DatabaseManager()
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.segmentation = CustomerSegmentation()
        self.recommendation_engine = RecommendationEngine()
        self.campaign_generator = CampaignGenerator()
        
        # Step 2: Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_period' not in st.session_state:
            st.session_state.current_period = "30d"
    
    def load_data(self):
        # Step 1: Load data from database
        try:
            self.customers_df = self.db_manager.get_customer_data()
            self.transactions_df = self.db_manager.get_transaction_data()
            self.campaigns_df = self.db_manager.get_campaign_data()

            # Step 2: Create customer features if not exists
            if not self.customers_df.empty and not self.transactions_df.empty:
                # Ensure transaction_date is datetime for all downstream math
                if self.transactions_df['transaction_date'].dtype == 'O':
                    self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'], errors='coerce')

                # Merge segment_name into transactions for segment filtering
                if 'segment_name' not in self.transactions_df.columns and 'segment_name' in self.customers_df.columns:
                    self.transactions_df = self.transactions_df.merge(
                        self.customers_df[['customer_id', 'segment_name']], on='customer_id', how='left'
                    )
                self.customer_features_df = self.data_preprocessor.create_customer_features(self.transactions_df)

                # Step 3: Perform segmentation if not already done
                if 'segment_name' not in self.customers_df.columns:
                    self.perform_segmentation()
                
                st.session_state.data_loaded = True
                self.logger.info("Data loaded successfully")
            else:
                st.warning("No data available. Please run the ETL pipeline first.")
                
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
    
    def perform_segmentation(self):
        # Step 1: Perform customer segmentation
        try:
            segmentation_results = self.segmentation.perform_segmentation(self.customer_features_df)
            
            # Step 2: Update database with segmentation results
            segments_df = segmentation_results[['customer_id', 'segment_id', 'segment_name']].copy()
            segments_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            self.db_manager.update_customer_segments(segments_df)
            
            # Step 3: Update local data
            self.customers_df = self.db_manager.get_customer_data()
            
            self.logger.info("Customer segmentation completed")
            
        except Exception as e:
            self.logger.error(f"Error performing segmentation: {str(e)}")
            st.error(f"Error performing segmentation: {str(e)}")
    
    def render_dashboard(self):
        # Step 1: Load data if not already loaded
        if not st.session_state.data_loaded:
            self.load_data()
        # Safety guard: ensure required attributes exist and contain data
        if not hasattr(self, 'transactions_df') or not hasattr(self, 'customers_df'):
            st.warning("No data available. Please run the ETL pipeline first.")
            return
        if self.transactions_df is None or self.customers_df is None:
            st.warning("No data available. Please run the ETL pipeline first.")
            return
        if getattr(self.transactions_df, 'empty', True) or getattr(self.customers_df, 'empty', True):
            st.warning("No data available. Please run the ETL pipeline first.")
            return
        
        # Step 2: Render sidebar
        render_sidebar()
        
        # Step 3: Main dashboard layout
        st.title("📊 Customer Spending Insights Dashboard")
        
        # Step 4: Render filters
        date_range, selected_categories, selected_segments = render_filters(
            self.transactions_df, self.customers_df
        )
        
        # Step 5: Apply filters
        filtered_transactions = self.apply_filters(
            self.transactions_df, date_range, selected_categories, selected_segments
        )
        
        # Step 6: Handle quick actions from sidebar (segmentation/export/report)
        self._handle_quick_actions(filtered_transactions)

        # Step 7: Render KPI cards
        kpi_data = self.calculate_kpis(filtered_transactions, date_range)
        render_kpi_cards(kpi_data)
        
        # Step 8: Render charts
        col1, col2 = st.columns(2)
        
        with col1:
            render_spending_chart(filtered_transactions)
            render_category_chart(filtered_transactions)
        
        with col2:
            render_segment_chart(self.customers_df)
            self.render_recommendations_chart()
        
        # Step 9: Render data tables
        self.render_data_tables(filtered_transactions)

    def _handle_quick_actions(self, filtered_transactions: pd.DataFrame):
        """Respond to sidebar actions using session flags."""
        # Run segmentation
        if st.session_state.get('run_segmentation'):
            try:
                self.perform_segmentation()
                st.success("Segmentation completed and database updated.")
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
            finally:
                st.session_state.run_segmentation = False

        # Export data
        if st.session_state.get('export_data'):
            try:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_dir = os.path.join('data', 'exports')
                os.makedirs(export_dir, exist_ok=True)

                # Save CSVs to disk
                tx_path = os.path.join(export_dir, f'transactions_{ts}.csv')
                cust_path = os.path.join(export_dir, f'customers_{ts}.csv')
                camp_path = os.path.join(export_dir, f'campaigns_{ts}.csv')
                self.transactions_df.to_csv(tx_path, index=False)
                self.customers_df.to_csv(cust_path, index=False)
                self.campaigns_df.to_csv(camp_path, index=False)

                st.success(f"Data exported to: {export_dir}")

                # Offer quick downloads (filtered transactions too)
                with st.expander("Download exports"):
                    # In-memory downloads
                    def _df_to_buffer(df: pd.DataFrame) -> io.BytesIO:
                        buf = io.BytesIO()
                        df.to_csv(buf, index=False)
                        buf.seek(0)
                        return buf

                    st.download_button("Download Transactions (filtered)", _df_to_buffer(filtered_transactions), file_name=f"transactions_filtered_{ts}.csv", mime="text/csv")
                    st.download_button("Download Transactions (all)", _df_to_buffer(self.transactions_df), file_name=f"transactions_{ts}.csv", mime="text/csv")
                    st.download_button("Download Customers", _df_to_buffer(self.customers_df), file_name=f"customers_{ts}.csv", mime="text/csv")
                    st.download_button("Download Campaigns", _df_to_buffer(self.campaigns_df), file_name=f"campaigns_{ts}.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Export failed: {e}")
            finally:
                st.session_state.export_data = False

        # Generate simple report
        if st.session_state.get('generate_report'):
            try:
                total_rev = float(filtered_transactions['amount'].sum()) if not filtered_transactions.empty else 0.0
                tx_count = int(len(filtered_transactions))
                uniq_cust = int(filtered_transactions['customer_id'].nunique()) if not filtered_transactions.empty else 0
                top_cat = (
                    filtered_transactions.groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
                    if not filtered_transactions.empty else pd.Series(dtype=float)
                )

                md = [
                    "# Customer Insights Report",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "## KPIs",
                    f"- Total Revenue: ${total_rev:,.2f}",
                    f"- Transactions: {tx_count:,}",
                    f"- Unique Customers: {uniq_cust:,}",
                    "",
                    "## Top Categories (by revenue)",
                ]
                for cat, val in top_cat.items():
                    md.append(f"- {cat}: ${val:,.2f}")
                report_md = "\n".join(md)

                st.success("Report generated.")
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report_md,
                    file_name="customer_insights_report.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")
            finally:
                st.session_state.generate_report = False
    
    def apply_filters(self, transactions_df: pd.DataFrame, date_range: tuple, 
                     categories: list, segments: list) -> pd.DataFrame:
        # Step 1: Apply date filter
        if date_range:
            start_date, end_date = date_range
            transactions_df = transactions_df[
                (transactions_df['transaction_date'] >= start_date) &
                (transactions_df['transaction_date'] <= end_date)
            ]
        
        # Step 2: Apply category filter
        if categories:
            transactions_df = transactions_df[transactions_df['category'].isin(categories)]
        
        # Step 3: Apply segment filter
        if segments:
            transactions_df = transactions_df[transactions_df['segment_name'].isin(segments)]
        
        return transactions_df
    
    def calculate_kpis(self, transactions_df: pd.DataFrame, date_range: Optional[tuple]) -> Dict[str, Any]:
        # Step 1: Calculate key performance indicators
        if transactions_df.empty:
            return {
                'total_revenue': 0,
                'total_transactions': 0,
                'avg_transaction_value': 0,
                'unique_customers': 0,
                'revenue_change': 0,
                'transaction_change': 0
            }
        
        # Step 2: Current period KPIs
        current_revenue = transactions_df['amount'].sum()
        current_transactions = len(transactions_df)
        current_avg_value = transactions_df['amount'].mean()
        current_customers = transactions_df['customer_id'].nunique()
        
        # Step 3: Previous period comparison
        if date_range:
            start_date, end_date = date_range
            period_days = (end_date - start_date).days
            prev_start = start_date - timedelta(days=period_days)
            prev_end = start_date
            
            prev_transactions = self.transactions_df[
                (self.transactions_df['transaction_date'] >= prev_start) &
                (self.transactions_df['transaction_date'] < prev_end)
            ]
            
            prev_revenue = prev_transactions['amount'].sum()
            prev_transaction_count = len(prev_transactions)
            
            revenue_change = ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            transaction_change = ((current_transactions - prev_transaction_count) / prev_transaction_count * 100) if prev_transaction_count > 0 else 0
        else:
            revenue_change = 0
            transaction_change = 0
        
        return {
            'total_revenue': current_revenue,
            'total_transactions': current_transactions,
            'avg_transaction_value': current_avg_value,
            'unique_customers': current_customers,
            'revenue_change': revenue_change,
            'transaction_change': transaction_change
        }
    
    def render_recommendations_chart(self):
        # Step 1: Render recommendations chart
        st.subheader("🎯 Top Recommendations")
        
        if not self.transactions_df.empty and not self.customer_features_df.empty:
            # Prepare recommendation engine (create matrix and similarity if needed)
            try:
                if getattr(self.recommendation_engine, 'customer_item_matrix', None) is None:
                    self.recommendation_engine.create_customer_item_matrix(self.transactions_df)
                if getattr(self.recommendation_engine, 'similarity_matrix', None) is None:
                    self.recommendation_engine.calculate_customer_similarity()
            except Exception as e:
                st.warning(f"Recommendations not available: {e}")
                return
            # Step 2: Get top customers for recommendations
            top_customers = self.customer_features_df.nlargest(5, 'total_spent')
            
            # Step 3: Generate recommendations
            recommendations_data = []
            for _, customer in top_customers.iterrows():
                customer_id = customer['customer_id']
                recs = self.recommendation_engine.generate_category_recommendations(customer_id, n_recommendations=3)
                
                for rec in recs:
                    recommendations_data.append({
                        'customer_id': customer_id,
                        'category': rec['category'],
                        'score': rec['score']
                    })
            
            if recommendations_data:
                recs_df = pd.DataFrame(recommendations_data)
                
                # Step 4: Create chart
                fig = px.bar(
                    recs_df.groupby('category')['score'].mean().reset_index(),
                    x='category',
                    y='score',
                    title="Top Recommended Categories",
                    color='score',
                    color_continuous_scale='viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Recommendation Score",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recommendations available")
        else:
            st.info("Insufficient data for recommendations")
    
    def render_data_tables(self, transactions_df: pd.DataFrame):
        # Step 1: Render data tables section
        st.subheader("📋 Data Tables")
        
        tab1, tab2, tab3 = st.tabs(["Transactions", "Customers", "Campaigns"])
        
        with tab1:
            if not transactions_df.empty:
                st.dataframe(
                    transactions_df[['transaction_date', 'customer_id', 'amount', 'category', 'merchant']].head(100),
                    use_container_width=True
                )
            else:
                st.info("No transaction data available")
        
        with tab2:
            if not self.customers_df.empty:
                st.dataframe(
                    self.customers_df[['customer_id', 'first_name', 'last_name', 'email', 'segment_name']].head(100),
                    use_container_width=True
                )
            else:
                st.info("No customer data available")
        
        with tab3:
            if not self.campaigns_df.empty:
                # Show discount column if present under either name
                discount_col = 'discount_percentage' if 'discount_percentage' in self.campaigns_df.columns else (
                    'discount_amount' if 'discount_amount' in self.campaigns_df.columns else None
                )
                display_cols = ['campaign_id', 'customer_id', 'campaign_type', 'status']
                if discount_col:
                    display_cols.insert(3, discount_col)
                st.dataframe(self.campaigns_df[display_cols].head(100), use_container_width=True)
            else:
                st.info("No campaign data available")

        # Step 9: Render model prediction demo
        self.render_model_prediction_demo()

    def render_model_prediction_demo(self):
        st.subheader("🤖 Model Prediction Demo")
        st.caption("Send a sample feature vector to the API Gateway which forwards to the ML Model Server.")

        gateway_url = os.getenv("GATEWAY_URL", "http://localhost:8002")
        colA, colB = st.columns([3, 1])
        with colA:
            features_text = st.text_input(
                "Features (comma-separated numbers)",
                value="1.0, 2.0, 3.0"
            )
        with colB:
            predict_btn = st.button("Predict", type="primary")

        if predict_btn:
            try:
                features = [float(x.strip()) for x in features_text.split(',') if x.strip()]
                resp = requests.post(f"{gateway_url}/predict", json={"features": features}, timeout=10)
                if resp.ok:
                    st.success(resp.json())
                else:
                    st.error(f"Gateway error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

def init_components():
    """Initialize and return commonly used components."""
    try:
        db_manager = DatabaseManager()
        data_loader = DataLoader()
        segmentation = CustomerSegmentation()
        campaign_generator = CampaignGenerator()
        return data_loader, db_manager, segmentation, campaign_generator
    except Exception:
        return None, None, None, None


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    """Load core tables from the database for dashboard pages."""
    _, db_manager, _, _ = init_components()
    if not db_manager:
        return None, None, None, None
    try:
        customers_df = db_manager.get_customer_data()
        transactions_df = db_manager.get_transaction_data()
        segments_df = db_manager.query_to_dataframe("SELECT * FROM customer_segments")
        campaigns_df = db_manager.get_campaign_data()
        # Ensure datetime types
        if transactions_df is not None and not transactions_df.empty:
            transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'], errors='coerce')
        if campaigns_df is not None and not campaigns_df.empty and 'created_date' in campaigns_df.columns:
            campaigns_df['created_date'] = pd.to_datetime(campaigns_df['created_date'], errors='coerce')
        return customers_df, transactions_df, segments_df, campaigns_df
    except Exception:
        return None, None, None, None


def main_dashboard():
    """Render the original main dashboard using DashboardApp class."""
    app = DashboardApp()
    app.render_dashboard()


def customer_analysis_page():
    """Customer analysis page."""
    st.title("🔍 Customer Analysis")
    customers_df, transactions_df, segments_df, campaigns_df = load_dashboard_data()
    if customers_df is None or customers_df.empty:
        st.warning("No data available")
        return
    customer_ids = customers_df['customer_id'].tolist()
    selected_customer = st.selectbox("Select Customer", customer_ids)
    if not selected_customer:
        return
    customer_data = customers_df[customers_df['customer_id'] == selected_customer].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Profile")
        st.write(f"**Name:** {customer_data.get('first_name','')} {customer_data.get('last_name','')}")
        st.write(f"**Email:** {customer_data.get('email','')}")
        st.write(f"**Age:** {customer_data.get('age','')}")
        st.write(f"**Location:** {customer_data.get('location','')}")
        st.write(f"**Registration:** {customer_data.get('registration_date','')}")
    with col2:
        if segments_df is not None and not segments_df.empty:
            customer_segment = segments_df[segments_df['customer_id'] == selected_customer]
            if not customer_segment.empty:
                st.subheader("Customer Segment")
                st.write(f"**Segment:** {customer_segment.iloc[0]['segment_name']}")
    if transactions_df is not None and not transactions_df.empty:
        cust_tx = transactions_df[transactions_df['customer_id'] == selected_customer]
        if not cust_tx.empty:
            st.subheader("Transaction History")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Spent", f"${cust_tx['amount'].sum():.2f}")
            with m2:
                st.metric("Transactions", int(len(cust_tx)))
            with m3:
                st.metric("Avg Amount", f"${cust_tx['amount'].mean():.2f}")
            fig = px.line(cust_tx.sort_values('transaction_date'), x='transaction_date', y='amount', title="Transaction History")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Recent Transactions")
            st.dataframe(cust_tx.sort_values('transaction_date', ascending=False).head(10), use_container_width=True)


def campaign_management_page():
    """Campaign management page."""
    st.title("🎯 Campaign Management")
    _, _, _, campaign_generator = init_components()
    if campaign_generator is None:
        st.error("Failed to initialize campaign components")
        return
    customers_df, transactions_df, segments_df, campaigns_df = load_dashboard_data()
    if customers_df is None or customers_df.empty:
        st.warning("No data available")
        return
    tab1, tab2, tab3 = st.tabs(["Generate Campaigns", "Active Campaigns", "Campaign Analytics"])
    with tab1:
        st.subheader("Generate New Campaigns")
        with st.form("campaign_form"):
            campaign_type = st.selectbox("Campaign Type", ["loyalty_reward", "win_back", "cross_sell", "seasonal_promotion"])
            if segments_df is not None and not segments_df.empty:
                available_segments = segments_df['segment_name'].unique().tolist()
                target_segments = st.multiselect("Target Segments", available_segments)
            else:
                target_segments = []
            budget = st.number_input("Budget", min_value=100, max_value=50000, value=5000)
            submitted = st.form_submit_button("Generate Campaigns")
            if submitted and target_segments:
                try:
                    all_campaigns = []
                    for segment in target_segments:
                        seg_customers = customers_df[customers_df['customer_id'].isin(segments_df[segments_df['segment_name'] == segment]['customer_id'])]
                        if not seg_customers.empty:
                            cdf = campaign_generator.generate_campaigns_for_segment(segment, seg_customers, seg_customers)
                            if not cdf.empty:
                                all_campaigns.extend(cdf.to_dict('records'))
                    if all_campaigns:
                        optimization = campaign_generator.optimize_campaign_portfolio(pd.DataFrame(all_campaigns), budget)
                        sel = optimization['selected_campaigns']
                        st.success(f"Generated {len(sel)} optimized campaigns")
                        st.write(f"**Budget Used:** ${optimization['total_budget_used']:.2f}")
                        st.write(f"**Expected ROI:** {optimization['expected_roi']:.2f}")
                        if sel:
                            st.dataframe(pd.DataFrame(sel), use_container_width=True)
                    else:
                        st.warning("No campaigns generated")
                except Exception as e:
                    st.error(f"Error generating campaigns: {e}")
    with tab2:
        st.subheader("Active Campaigns")
        if campaigns_df is not None and not campaigns_df.empty:
            active = campaigns_df[campaigns_df['status'] == 'active'] if 'status' in campaigns_df.columns else campaigns_df
            if not active.empty:
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("Active Campaigns", int(len(active)))
                with c2:
                    total_discount = float(active.get('discount_amount', pd.Series([0]*len(active))).sum())
                    st.metric("Total Discount", f"${total_discount:.2f}")
                with c3:
                    avg_discount = float(active.get('discount_amount', pd.Series([0]*len(active))).mean())
                    st.metric("Avg Discount", f"${avg_discount:.2f}")
                with c4:
                    c_types = int(active['campaign_type'].nunique()) if 'campaign_type' in active.columns else 0
                    st.metric("Campaign Types", c_types)
                if 'campaign_type' in active.columns:
                    counts = active['campaign_type'].value_counts()
                    fig = px.bar(x=counts.values, y=counts.index, orientation='h', title="Active Campaigns by Type")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(active, use_container_width=True)
            else:
                st.info("No active campaigns")
        else:
            st.info("No campaign data available")
    with tab3:
        st.subheader("Campaign Analytics")
        if campaigns_df is not None and not campaigns_df.empty:
            col1,col2 = st.columns(2)
            if 'status' in campaigns_df.columns:
                with col1:
                    status_counts = campaigns_df['status'].value_counts()
                    fig = px.pie(values=status_counts.values, names=status_counts.index, title="Campaigns by Status")
                    st.plotly_chart(fig, use_container_width=True)
            if 'campaign_type' in campaigns_df.columns:
                with col2:
                    type_counts = campaigns_df['campaign_type'].value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Campaigns by Type")
                    st.plotly_chart(fig, use_container_width=True)
            if 'created_date' in campaigns_df.columns:
                timeline = campaigns_df.groupby(campaigns_df['created_date'].dt.date).size().reset_index()
                timeline.columns = ['date','count']
                fig = px.line(timeline, x='date', y='count', title="Campaign Creation Timeline")
                st.plotly_chart(fig, use_container_width=True)


def segmentation_page():
    """Customer segmentation page."""
    st.title("📊 Customer Segmentation")
    _, _, segmentation, _ = init_components()
    if segmentation is None:
        st.error("Failed to initialize segmentation components")
        return
    customers_df, transactions_df, segments_df, _ = load_dashboard_data()
    if customers_df is None or transactions_df is None:
        st.warning("Insufficient data for segmentation")
        return
    tab1, tab2, tab3 = st.tabs(["Current Segments", "Segmentation Analysis", "Re-segment"]) 
    with tab1:
        st.subheader("Current Customer Segments")
        if segments_df is not None and not segments_df.empty:
            counts = segments_df['segment_name'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=counts.values, names=counts.index, title="Segment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.subheader("Segment Details")
                for seg, cnt in counts.items():
                    st.write(f"**{seg}:** {cnt:,} customers ({(cnt/len(segments_df))*100:.1f}%)")
            st.dataframe(segments_df, use_container_width=True)
        else:
            st.info("No segmentation data available")
    with tab2:
        st.subheader("Segmentation Analysis")
        if segments_df is not None and not segments_df.empty and not transactions_df.empty:
            try:
                from src.data.data_preprocessor import DataPreprocessor
                pre = DataPreprocessor()
                feats = pre.create_customer_features(transactions_df)
                if not feats.empty:
                    seg_analysis = feats.merge(segments_df[['customer_id','segment_name']], on='customer_id', how='inner')
                    metrics = seg_analysis.groupby('segment_name').agg({
                        'total_spent': ['mean','median'],
                        'avg_transaction_amount': 'mean',
                        'transaction_count': 'mean',
                        'recency_days': 'mean'
                    }).round(2)
                    st.subheader("Segment Comparison")
                    st.dataframe(metrics, use_container_width=True)
                    fig = px.box(seg_analysis, x='segment_name', y='total_spent', title="Spending Distribution by Segment")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in segmentation analysis: {e}")
        else:
            st.info("Insufficient data for analysis")
    with tab3:
        st.subheader("Re-run Segmentation")
        with st.form("segmentation_form"):
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
            submitted = st.form_submit_button("Run Segmentation")
            if submitted:
                try:
                    from src.data.data_preprocessor import DataPreprocessor
                    pre = DataPreprocessor()
                    feats = pre.create_customer_features(transactions_df)
                    if not feats.empty:
                        with st.spinner("Running segmentation..."):
                            res = segmentation.perform_segmentation(feats, n_clusters)
                        if not res.empty:
                            st.success(f"Segmentation completed with {n_clusters} clusters")
                            counts = res['segment_name'].value_counts()
                            fig = px.pie(values=counts.values, names=counts.index, title="New Segment Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                            st.subheader("Segmentation Results (Sample)")
                            st.dataframe(res.head(20), use_container_width=True)
                        else:
                            st.error("Segmentation failed")
                    else:
                        st.error("Could not create customer features")
                except Exception as e:
                    st.error(f"Error during segmentation: {e}")


def main():
    """Main multipage application entry."""
    st.sidebar.title("CICOP Navigation")
    pages = {
        "🏠 Dashboard": main_dashboard,
        "👤 Customer Analysis": customer_analysis_page,
        "🎯 Campaign Management": campaign_management_page,
        "📊 Segmentation": segmentation_page,
    }
    selected = st.sidebar.selectbox("Choose a page", list(pages.keys()))
    if st.sidebar.button("🔄 Refresh Data"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()
    st.sidebar.subheader("System Status")
    try:
        _, dbm, _, _ = init_components()
        if dbm:
            stats = dbm.get_database_stats()
            st.sidebar.success("✅ Database Connected")
            st.sidebar.write(f"Customers: {stats.get('customers',0):,}")
            st.sidebar.write(f"Transactions: {stats.get('transactions',0):,}")
        else:
            st.sidebar.error("❌ Database Error")
    except Exception:
        st.sidebar.error("❌ System Error")
    # Render selected page
    pages[selected]()

if __name__ == "__main__":
    main() 