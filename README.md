# Customer Intelligence & Campaign Orchestration Platform (CICOP)

A next-generation customer analytics and intelligent campaign management system that leverages advanced machine learning, real-time data processing, and automated decision-making to drive customer engagement and revenue growth.

---

## ⚡ Local Quick Start (Windows)

This repo includes ready-to-run scripts to set up data and start all services locally using SQLite.

### 1) Create and load sample data (ETL)

```bat
scripts\run_etl.cmd
```

What it does:
- Ensures `DATABASE_URL=sqlite:///data/customer_insights.db`
- Runs `scripts/setup_database.py`
- Runs `scripts/generate_standalone_data.py`

### 2) Start all services

```bat
scripts\start_all_services.cmd
```

It launches in separate terminals:
- API (FastAPI/Uvicorn): http://localhost:8000
- ML Model Server: http://localhost:8001
- API Gateway: http://localhost:8002
- Streamlit Dashboard: http://localhost:8501

If you prefer to run manually:

```bat
set DATABASE_URL=sqlite:///data/customer_insights.db
set PYTHONPATH=%CD%

REM API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

REM ML server
uvicorn src.ml.model_server:app --host 0.0.0.0 --port 8001 --reload

REM API Gateway
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8002 --reload

REM Dashboard (in a separate terminal, from project root)
set PYTHONPATH=%CD%
streamlit run src\dashboard\app.py --server.port=8501 --server.address=0.0.0.0
```

### 3) Useful endpoints

- API root: `GET /` → status and docs link
- Health: `GET /health` → JSON with database stats
- Debug: `GET /debug` → environment/module diagnostics
- Docs: `GET /docs`

Protected endpoints (e.g., `/analytics/dashboard`) require auth. Obtain a token via `POST /auth/login` and send `Authorization: Bearer <token>`.

### 4) Troubleshooting

- Port in use (8000):
  - `netstat -ano | findstr :8000` then `taskkill /PID <pid> /F`
- Health returns error:
  - We added robust JSON serialization and graceful fallbacks. Restart API with:
    - `set DATABASE_URL=sqlite:///data/customer_insights.db`
    - `set PYTHONPATH=%CD%`
    - `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug`
- Streamlit shows 0.0.0.0 URL:
  - Open the dashboard at `http://localhost:8501` in your browser.
- Unicode logging error in scripts:
  - Our scripts set UTF-8 automatically. If running manually, set `PYTHONIOENCODING=utf-8`.

### 5) Scripts

- `scripts/run_etl.cmd` – initializes schema and generates sample data
- `scripts/start_all_services.cmd` – starts API, ML, Gateway, and Dashboard with UTF‑8 console

---

## ✅ Current Capabilities

These are implemented and working in this repository:

- **API (FastAPI)** in `src/api/main.py`
  - `GET /` root, `GET /health`, `GET /debug`, docs at `/docs`
  - Global exception handlers and detailed logging
  - Basic auth scaffolding; some routes protected via Bearer token

- **Data layer (SQLite)** via `src/data/database_manager.py`
  - Schema initialization, simple query/insert helpers
  - JSON-safe stats (native Python ints)

- **Dashboard (Streamlit)** in `src/dashboard/`
  - KPI cards and visualizations in `src/dashboard/components/`
  - Sidebar, filters, and charts

- **ML basics** in `src/ml/`
  - Customer features, basic segmentation, simple recommendation engine
  - Model utilities for basic reporting/persistence

- **Developer scripts** in `scripts/`
  - `run_etl.cmd` to initialize schema and generate sample data
  - `start_all_services.cmd` to launch API, ML, Gateway, Dashboard

- **Serialization reliability**
  - Robust NumPy/Pandas → JSON conversion via `src/utils/json_encoder.py`

---

## 🛣️ Roadmap (Planned / Not Yet Implemented)

Near-term:
- **Hardened auth & RBAC**: user roles/permissions and session management
- **Analytics endpoint**: JSON-safe `/analytics/dashboard` payloads; pagination
- **Test coverage**: unit/integration tests for data, API, and dashboard

Mid-term:
- **GraphQL support** and versioned REST structure (`v1/`, `v2/`)
- **A/B testing utilities** and campaign portfolio scoring
- **Production database** option and migrations (PostgreSQL)

Long-term (vision):
- **Real-time streaming** (Kafka, Spark Streaming) and event sourcing
- **AutoML/MLflow/Feature Store** for model lifecycle
- **Kubernetes/Terraform** deployment blueprints
- **Enterprise integrations** (CRM/marketing platforms/warehouses)

> Contributions toward any roadmap item are welcome. See the Contributing section below.

## 🚀 Advanced Features

> Note: The sections below describe the long‑term product vision. Some items are not yet implemented in this repository. See the "Current Capabilities" and "Roadmap" sections above for the accurate status.

### **AI-Powered Customer Intelligence**
- **Dynamic Segmentation**: Real-time clustering with adaptive algorithms (K-means, DBSCAN, hierarchical)
- **Behavioral Analytics**: Deep learning models for spending pattern recognition and churn prediction
- **Predictive Scoring**: ML-powered customer lifetime value and response probability models
- **Real-time Insights**: Streaming analytics with Apache Kafka integration capabilities

### **Intelligent Campaign Orchestration**
- **Multi-Channel Campaigns**: Automated campaign generation across email, SMS, push notifications, and social media
- **Dynamic Personalization**: Real-time content adaptation based on customer behavior and context
- **A/B Testing Engine**: Built-in experimentation framework for campaign optimization
- **ROI Optimization**: Advanced portfolio optimization with budget constraints and expected return modeling

### **Enterprise-Grade Architecture**
- **Microservices Ready**: Containerized architecture with Docker and Kubernetes support
- **Real-time Processing**: Apache Spark streaming for live data analytics
- **Scalable Database**: Support for PostgreSQL, MongoDB, and Redis with connection pooling
- **API-First Design**: RESTful APIs with GraphQL support and comprehensive documentation

### **Advanced Analytics Dashboard**
- **Interactive Visualizations**: D3.js powered charts with real-time updates
- **Custom Dashboards**: Drag-and-drop dashboard builder for business users
- **Mobile Responsive**: Progressive Web App with offline capabilities
- **Role-Based Access**: Multi-tenant architecture with granular permissions

## 🏗️ Enhanced Project Structure

```
customer_intelligence_platform/
├── src/
│   ├── core/              # Core business logic and domain models
│   ├── data/              # Data ingestion, processing, and storage
│   │   ├── streaming/     # Real-time data processing
│   │   ├── batch/         # Batch data processing
│   │   └── warehouse/     # Data warehouse integration
│   ├── ml/                # Machine learning and AI models
│   │   ├── models/        # Pre-trained and custom ML models
│   │   ├── training/      # Model training pipelines
│   │   └── inference/     # Real-time model serving
│   ├── rules/             # Business rules and decision engine
│   │   ├── engine/        # Rules execution engine
│   │   ├── builder/       # Visual rules builder
│   │   └── templates/     # Pre-built campaign templates
│   ├── dashboard/         # Web-based analytics dashboard
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Dashboard pages and views
│   │   └── widgets/       # Custom dashboard widgets
│   ├── api/               # REST and GraphQL APIs
│   │   ├── v1/            # API version 1
│   │   ├── v2/            # API version 2 (latest)
│   │   └── graphql/       # GraphQL schema and resolvers
│   ├── integrations/      # Third-party system integrations
│   │   ├── crm/           # CRM system connectors
│   │   ├── marketing/     # Marketing automation platforms
│   │   └── analytics/     # Analytics and BI tools
│   └── utils/             # Shared utilities and helpers
├── infrastructure/         # Infrastructure and deployment
│   ├── docker/            # Docker configurations
│   ├── kubernetes/        # K8s manifests and configs
│   ├── terraform/         # Infrastructure as code
│   └── monitoring/        # Monitoring and alerting
├── config/                # Configuration management
├── data/                  # Data storage and samples
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Comprehensive test suite
├── docs/                  # API documentation and guides
└── scripts/               # Automation and utility scripts
```

## 🛠️ Installation & Setup

### **Quick Start (Development)**
```bash
# Clone the repository
git clone https://github.com/your-org/customer-intelligence-platform.git
cd customer-intelligence-platform

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize the system
python scripts/setup_database.py
python scripts/load_sample_data.py
python scripts/start_services.py
```

### **Production Deployment**
```bash
# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Using Kubernetes
kubectl apply -f infrastructure/kubernetes/
```

## 📊 Usage Examples

### **Advanced Customer Segmentation**
```python
from src.ml.customer_segmentation import CustomerSegmentation
from src.ml.models import AdvancedClusteringModel

# Initialize with custom model
segmentation = CustomerSegmentation(
    model=AdvancedClusteringModel(
        algorithm='hdbscan',
        hyperparameters={'min_cluster_size': 10}
    )
)

# Perform real-time segmentation
segments = segmentation.perform_segmentation(
    customer_data,
    features=['spending_pattern', 'engagement_score', 'risk_profile'],
    real_time=True
)
```

### **Intelligent Campaign Orchestration**
```python
from src.rules.campaign_orchestrator import CampaignOrchestrator
from src.rules.templates import CampaignTemplate

# Create campaign orchestrator
orchestrator = CampaignOrchestrator()

# Generate multi-channel campaigns
campaigns = orchestrator.generate_campaigns(
    target_segments=['high_value', 'at_risk'],
    channels=['email', 'sms', 'push'],
    personalization_level='advanced',
    a_b_testing=True
)

# Optimize campaign portfolio
optimized = orchestrator.optimize_portfolio(
    campaigns,
    budget_constraint=50000,
    roi_target=0.15,
    risk_tolerance='moderate'
)
```

### **Real-time Analytics API**
```python
from src.api.client import CustomerIntelligenceClient

# Initialize API client
client = CustomerIntelligenceClient(api_key='your_key')

# Get real-time customer insights
insights = client.get_customer_insights(
    customer_id='CUST_001',
    include_predictions=True,
    include_recommendations=True
)

# Stream customer behavior events
for event in client.stream_customer_events():
    print(f"Customer {event.customer_id}: {event.action}")
```

## ⚙️ Configuration

### **Environment Variables**
```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/cicop
REDIS_URL=redis://localhost:6379

# ML Model Configuration
ML_MODEL_ENDPOINT=http://localhost:8000/predict
MODEL_UPDATE_INTERVAL=3600

# API Configuration
API_VERSION=v2
RATE_LIMIT=1000
CACHE_TTL=300
```

### **Advanced ML Configuration**
```yaml
# config/ml_config.yaml
models:
  customer_segmentation:
    algorithm: hdbscan
    hyperparameters:
      min_cluster_size: 10
      min_samples: 5
    auto_tuning: true
    ensemble_methods: [bagging, boosting]
  
  churn_prediction:
    algorithm: xgboost
    features: [recency, frequency, monetary, engagement]
    threshold: 0.7
    retrain_interval: 86400
```

## 🧪 Testing & Quality Assurance

### **Comprehensive Test Suite**
```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests
pytest tests/e2e/          # End-to-end tests

# Run with different configurations
pytest --config=test_config.yaml
```

### **Performance Testing**
```bash
# Load testing
locust -f tests/performance/locustfile.py

# Memory profiling
python -m memory_profiler scripts/performance_test.py
```

## 🚀 Advanced Capabilities

### **Real-time Data Processing**
- **Apache Kafka Integration**: Stream processing for real-time customer events
- **Apache Spark Streaming**: Distributed stream processing for large-scale data
- **Event Sourcing**: Complete audit trail of all customer interactions

### **Machine Learning Pipeline**
- **AutoML**: Automated model selection and hyperparameter tuning
- **Model Versioning**: MLflow integration for model lifecycle management
- **Feature Store**: Centralized feature management and serving
- **A/B Testing**: Statistical testing framework for model comparison

### **Enterprise Integrations**
- **CRM Systems**: Salesforce, HubSpot, Microsoft Dynamics
- **Marketing Platforms**: Mailchimp, SendGrid, Twilio
- **Analytics Tools**: Google Analytics, Mixpanel, Amplitude
- **Data Warehouses**: Snowflake, BigQuery, Redshift

## 📈 Performance & Scalability

- **Horizontal Scaling**: Support for 1M+ customers with sub-second response times
- **Caching Strategy**: Multi-level caching with Redis and in-memory caches
- **Database Optimization**: Connection pooling, query optimization, and indexing
- **Load Balancing**: Automatic load distribution across multiple instances

## 🔒 Security & Compliance

- **Data Encryption**: AES-256 encryption for data at rest and in transit
- **Access Control**: Role-based access control (RBAC) with OAuth 2.0
- **Audit Logging**: Comprehensive audit trail for compliance requirements
- **GDPR Compliance**: Built-in data privacy and right-to-be-forgotten features

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Workflow**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Submit a pull request with detailed description

### **Code Quality Standards**
- Follow PEP 8 style guidelines
- Maintain 90%+ test coverage
- Use type hints for all functions
- Document all public APIs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.cicop.com](https://docs.cicop.com)
- **Community**: [community.cicop.com](https://community.cicop.com)
- **Issues**: [GitHub Issues](https://github.com/your-org/customer-intelligence-platform/issues)
- **Email**: support@cicop.com

---

**Built with ❤️ by the CICOP Team** 