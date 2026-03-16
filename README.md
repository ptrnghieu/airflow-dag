# Run Airflow with Docker

This guide will help you run Airflow ArXiv Scraper using Docker on Windows (or any OS).

## Requirements

- Docker Desktop (installed and running)
- Docker Compose (included with Docker Desktop)
- Git Bash or WSL2 (to run .sh scripts)


### Start manually

```bash
# 1. Create a .env file (copy from template)
# Create a .env file with the following content:
```

Content of the `.env` file:
```env
# Airflow Admin User
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_ADMIN_EMAIL=admin@example.com

# Airflow configuration (Only for information)
AIRFLOW_UID=50000
AIRFLOW_IMAGE_NAME=apache/airflow:3.1.1rc1-python3.10

# Database 
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow
```

```bash
#2. Create necessary folders
mkdir logs
mkdir plugins
mkdir -p tmp/arxiv_data

#3. Build Docker image
docker compose build

# 4. Initialize database
docker compose up airflow-init

#5. Start services
docker compose up -d
```


### Important configurations have been set

```yaml
environment: 
# Executor - Allows running multiple tasks in parallel 
AIRFLOW__CORE__EXECUTOR: LocalExecutor 

# Database - PostgreSQL 
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow 

# Task SDK - Connect to API server 
AIRFLOW__TASK_SDK__BASE_URL: http://airflow-webserver:8080 
AIRFLOW__TASK_SDK__API_VERSION: '2025-10-27' 

# JWT Authentication - REQUIRED for Execution API 
AIRFLOW__API_AUTH__JWT_SECRET: 'my-super-secret-jwt-key-for-airflow-2025' 
AIRFLOW__EXECUTION_API__JWT_AUDIENCE: 'urn:airflow.apache.org:task' 
AIRFLOW__EXECUTION_API__JWT_EXPIRATION_TIME: '600'
```

## 🌐 Access Airflow

After successful startup:

- **Web UI**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin`

## 📋 Management commands

### View logs

```bash
# View logs of all services
docker compose logs -f

# View logs of a specific service
docker compose logs -f airflow-webserver
docker compose logs -f airflow-scheduler
```

### Stop Airflow

```bash
# Stop services (keep data)
docker compose down

# Or use script
bash docker-stop.sh
```

### Delete completely (including database)

```bash
docker compose down -v
```

### Restart

```bash
docker compose restart
```

### Access MinIO Console
- URL: http://localhost:9001
- Username: minioadmin
- Password: minioadmin123

## Access MongoDB

### Using MongoDB Compass (GUI)
```
Connection String: mongodb://admin:admin123@localhost:27017/
Database: arxiv_db
Collection: papers
```
## Using DAGs

1. Go to http://localhost:8080
2. Login with `admin`/`admin`
3. Find DAG `arxiv_paper_scraper`
4. Turn on the toggle to activate DAG
5. Click "Trigger DAG" to run manually or wait for it to run on a schedule


## 🔍 Data Cleaning Details

### Processing Steps:

1. **Remove Duplicates**: Remove papers with the same ID

2. **String Normalization**:
- Remove extra spaces
- Remove invalid special characters
- Trim whitespace
3. **Missing Values**: Replace empty strings with None
4. **URL Validation**: Check the validity of PDF URLs
5. **Date Formatting**: Ensure the format is YYYY-MM-DD
6. **Critical Fields Check**: Remove papers missing ID or title
7. **Quality Flag**: Add `data_quality` flag for tracking

## 📊 View output data

Data is saved in the folder `tmp/arxiv_data/`:
```bash
# List files
ls -la tmp/arxiv_data/

# View CSV
cat tmp/arxiv_data/arxiv_papers_*.csv
```

Check the logs of each task in Airflow UI:
- `clean_data`: View statistics about data cleaning
- `save_to_mongodb`: View the number of papers inserted/updated

Using MongoDB Compass to check data
## Troubleshooting


### Task failed with "Invalid auth token: Signature verification failed"

**Cause:** JWT secrets are not synchronized between scheduler and webserver

**Solution:**

```bash
# MUST down and re-up, DO NOT just restart
docker-compose down
docker-compose up -d

# Verify JWT is synchronized:
docker exec airflowsimple-airflow-scheduler-1 python -c \
"from airflow.configuration import conf; print(conf.get('api_auth', 'jwt_secret')[:20])"

docker exec airflowsimple-airflow-webserver-1 python -c \
"from airflow.configuration import conf; print(conf.get('api_auth', 'jwt_secret')[:20])"

# The two outputs must be THE SAME
```

### Error: "Cannot connect to the Docker daemon"

Make sure Docker Desktop is running.

### Error: "Port 8080 already in use"

Change port in `docker-compose.yaml`:
```yaml
ports:
- "8081:8080" # Change 8080 to 8081
```

### Error: Don't see DAG in UI (Example for training)
```bash
docker exec airflowsimple-airflow-scheduler-1 python /opt/airflow/dags/arxiv_training_dag.py

docker exec airflowsimple-airflow-scheduler-1 airflow dags reserialize
```


### Error: "Permission denied" on Linux/macOS

```bash
# Set permissions for script
chmod +x docker-start.sh docker-stop.sh
```

### DAG not appearing in UI

```bash
# Check logs
docker compose logs airflow-scheduler

# Restart scheduler
docker compose restart airflow-scheduler
```

### Want to delete database and start over

```bash
docker compose down -v
docker compose up airflow-init
docker compose up -d
```

## Security

**Important Important**: Change the admin password in the `.env` file before deploying to production:

```env
AIRFLOW_ADMIN_USERNAME=your_username
AIRFLOW_ADMIN_PASSWORD=strong_password_here
AIRFLOW_ADMIN_EMAIL=your_email@domain.com
```

Then rebuild:
```bash
docker compose down -v
docker compose up airflow-init
docker compose up -d
```

## 🤖 Machine Learning Training

### Training DAG: `arxiv_category_trainer`

**Purpose**: Train ML model to predict paper categories from title and abstract.

**Features**:
- **Multi-label Classification**: Predict multiple categories per paper
- **Algorithm**: OneVsRestClassifier(LogisticRegression) + TF-IDF
- **Data Source**: MongoDB papers collection
- **Output**: Trained model artifacts saved locally

**Quick Start**:
```bash
# Trigger training DAG
docker exec airflowsimple-airflow-scheduler-1 airflow dags trigger arxiv_category_trainer

# Check status
docker exec airflowsimple-airflow-scheduler-1 airflow dags state arxiv_category_trainer
```

**Model Artifacts**:
- `model.joblib` - Trained classifier
- `vectorizer.joblib` - TF-IDF vectorizer  
- `label_encoder.joblib` - Multi-label binarizer

**Usage**:
```python
import joblib

# Load model
model = joblib.load('tmp/ml_training/models/model.joblib')
vectorizer = joblib.load('tmp/ml_training/models/vectorizer.joblib')
mlb = joblib.load('tmp/ml_training/models/label_encoder.joblib')
```

**Requirements**:
- Minimum 10 papers in MongoDB
- Dependencies: `scikit-learn`, `joblib`
- Training time: ~1-2 minutes

### For test model run in local:
```bash
python inference.py
```