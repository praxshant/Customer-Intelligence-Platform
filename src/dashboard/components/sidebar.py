# sidebar.py
# Purpose: Sidebar component for navigation and settings

import streamlit as st
from datetime import datetime
from typing import Dict, Any

def render_sidebar():
    # Step 1: Sidebar header
    st.sidebar.title("🎯 Customer Insights")
    st.sidebar.markdown("---")
    
    # Step 2: Navigation menu
    render_navigation()
    
    # Step 3: Quick actions
    render_quick_actions()
    
    # Step 4: Settings
    render_settings()
    
    # Step 5: System info
    render_system_info()

def render_navigation():
    # Step 1: Navigation menu
    st.sidebar.subheader("📋 Navigation")
    
    # Step 2: Main dashboard (single-page app - no external page switch)
    if st.sidebar.button("🏠 Dashboard", width='stretch'):
        st.toast("Already on Dashboard", icon="🏠")
    
    # Step 3: Customer analysis (placeholder navigation)
    if st.sidebar.button("👥 Customer Analysis", width='stretch'):
        st.toast("Customer Analysis view not yet implemented.")
    
    # Step 4: Campaign management (placeholder navigation)
    if st.sidebar.button("🎯 Campaign Management", width='stretch'):
        st.toast("Campaign Management view not yet implemented.")
    
    # Step 5: Reports (placeholder navigation)
    if st.sidebar.button("📊 Reports", width='stretch'):
        st.toast("Reports view not yet implemented.")
    
    # Step 6: Settings (placeholder navigation)
    if st.sidebar.button("⚙️ Settings", width='stretch'):
        st.toast("Settings view not yet implemented.")

def render_quick_actions():
    # Step 1: Quick actions section
    st.sidebar.subheader("⚡ Quick Actions")
    
    # Step 2: Refresh data
    if st.sidebar.button("🔄 Refresh Data", width='stretch'):
        if 'data_loaded' in st.session_state:
            st.session_state.data_loaded = False
        st.rerun()
    
    # Step 3: Export data
    if st.sidebar.button("📤 Export Data", width='stretch'):
        st.session_state.export_data = True
        st.toast("Exporting data...", icon="📤")
    
    # Step 4: Generate report
    if st.sidebar.button("📋 Generate Report", width='stretch'):
        st.session_state.generate_report = True
        st.toast("Generating report...", icon="📋")
    
    # Step 5: Run segmentation
    if st.sidebar.button("🎯 Run Segmentation", width='stretch'):
        st.session_state.run_segmentation = True
        st.toast("Running segmentation...", icon="🎯")

def render_settings():
    # Step 1: Settings section
    st.sidebar.subheader("⚙️ Settings")
    
    # Step 2: Theme selection
    theme = st.sidebar.selectbox(
        "Theme",
        ["Light", "Dark"],
        index=0
    )
    
    # Step 3: Data refresh interval
    refresh_interval = st.sidebar.selectbox(
        "Auto Refresh Interval",
        ["Off", "30 seconds", "1 minute", "5 minutes", "15 minutes"],
        index=0
    )
    
    # Step 4: Chart type preference
    chart_type = st.sidebar.selectbox(
        "Preferred Chart Type",
        ["Plotly", "Altair", "Matplotlib"],
        index=0
    )
    
    # Step 5: Save settings
    if st.sidebar.button("💾 Save Settings", width='stretch'):
        st.sidebar.success("Settings saved!")

def render_system_info():
    # Step 1: System information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Info")
    
    # Step 2: Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.write(f"🕒 **Time:** {current_time}")
    
    # Step 3: Data status
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        st.sidebar.write("✅ **Data Status:** Loaded")
    else:
        st.sidebar.write("❌ **Data Status:** Not Loaded")
    
    # Step 4: Version info
    st.sidebar.write("📦 **Version:** 1.0.0")
    
    # Step 5: Help and support
    st.sidebar.markdown("---")
    st.sidebar.subheader("❓ Help & Support")
    
    if st.sidebar.button("📖 Documentation", width='stretch'):
        st.sidebar.info("Documentation - Coming Soon!")
    
    if st.sidebar.button("🐛 Report Bug", width='stretch'):
        st.sidebar.info("Bug reporting - Coming Soon!")
    
    if st.sidebar.button("💬 Contact Support", width='stretch'):
        st.sidebar.info("Support contact - Coming Soon!")

def render_data_summary(data_stats: Dict[str, Any]):
    # Step 1: Data summary in sidebar
    st.sidebar.subheader("📊 Data Summary")
    
    # Step 2: Display data statistics
    if data_stats:
        st.sidebar.write(f"👥 **Customers:** {data_stats.get('customers', 0):,}")
        st.sidebar.write(f"🛒 **Transactions:** {data_stats.get('transactions', 0):,}")
        st.sidebar.write(f"🎯 **Campaigns:** {data_stats.get('campaigns', 0):,}")
        st.sidebar.write(f"🏷️ **Segments:** {data_stats.get('segments', 0):,}")
    else:
        st.sidebar.write("No data available")

def render_quick_metrics(metrics: Dict[str, Any]):
    # Step 1: Quick metrics in sidebar
    st.sidebar.subheader("📈 Quick Metrics")
    
    # Step 2: Display key metrics
    if metrics:
        st.sidebar.metric(
            "💰 Total Revenue",
            f"${metrics.get('total_revenue', 0):,.0f}"
        )
        
        st.sidebar.metric(
            "🛒 Transactions",
            f"{metrics.get('total_transactions', 0):,}"
        )
        
        st.sidebar.metric(
            "👥 Customers",
            f"{metrics.get('unique_customers', 0):,}"
        )
    else:
        st.sidebar.write("No metrics available")

def render_recent_activity():
    # Step 1: Recent activity section
    st.sidebar.subheader("🕒 Recent Activity")
    
    # Step 2: Mock recent activities
    activities = [
        {"time": "2 min ago", "action": "Data refreshed", "icon": "🔄"},
        {"time": "5 min ago", "action": "Segmentation updated", "icon": "🎯"},
        {"time": "10 min ago", "action": "Campaign generated", "icon": "📢"},
        {"time": "15 min ago", "action": "Report exported", "icon": "📤"}
    ]
    
    for activity in activities:
        st.sidebar.write(f"{activity['icon']} {activity['action']}")
        st.sidebar.caption(f"_{activity['time']}_")

def render_notifications():
    # Step 1: Notifications section
    st.sidebar.subheader("🔔 Notifications")
    
    # Step 2: Mock notifications
    notifications = [
        {"type": "info", "message": "New data available", "icon": "ℹ️"},
        {"type": "warning", "message": "Segmentation needs update", "icon": "⚠️"},
        {"type": "success", "message": "Campaign sent successfully", "icon": "✅"}
    ]
    
    for notification in notifications:
        if notification['type'] == 'info':
            st.sidebar.info(f"{notification['icon']} {notification['message']}")
        elif notification['type'] == 'warning':
            st.sidebar.warning(f"{notification['icon']} {notification['message']}")
        elif notification['type'] == 'success':
            st.sidebar.success(f"{notification['icon']} {notification['message']}")

def render_user_profile():
    # Step 1: User profile section
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 User Profile")
    
    # Step 2: Mock user info
    st.sidebar.write("**Name:** Analytics User")
    st.sidebar.write("**Role:** Data Analyst")
    st.sidebar.write("**Last Login:** Today")
    
    # Step 3: Profile actions
    if st.sidebar.button("👤 Edit Profile", width='stretch'):
        st.sidebar.info("Profile editing - Coming Soon!")
    
    if st.sidebar.button("🚪 Logout", width='stretch'):
        st.sidebar.info("Logout functionality - Coming Soon!") 