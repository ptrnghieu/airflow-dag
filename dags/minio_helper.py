"""
MinIO helper functions for data storage
Replaces XCom for inter-task data transfer
"""
import json
import logging
from io import BytesIO
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinioClient:
    """
    MinIO client wrapper for storing and retrieving data between Airflow tasks.
    """
    
    def __init__(
        self, 
        endpoint="minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        secure=False
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO server endpoint
            access_key: Access key
            secret_key: Secret key
            secure: Use HTTPS if True
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = "airflow-data"
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist - with retry logic"""
        import time
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if not self.client.bucket_exists(self.bucket_name):
                    self.client.make_bucket(self.bucket_name)
                    logger.info(f"Created MinIO bucket: {self.bucket_name}")
                else:
                    logger.info(f"MinIO bucket already exists: {self.bucket_name}")
                return  # Success
            except S3Error as e:
                if attempt < max_retries - 1:
                    logger.warning(f"MinIO not ready (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Error creating bucket after {max_retries} attempts: {str(e)}")
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to MinIO after {max_retries} attempts: {str(e)}")
                    raise e
    
    def save_json(self, object_name: str, data: dict | list) -> bool:
        """
        Save Python object (dict/list) as JSON to MinIO.
        
        Args:
            object_name: Name of the object to store
            data: Python dict or list to save
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert to JSON string
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            json_bytes = json_data.encode('utf-8')
            
            # Upload to MinIO
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=BytesIO(json_bytes),
                length=len(json_bytes),
                content_type='application/json'
            )
            
            logger.info(f"Saved data to MinIO: {object_name} ({len(json_bytes)} bytes)")
            return True
            
        except S3Error as e:
            logger.error(f"Error saving to MinIO: {str(e)}")
            raise e
    
    def load_json(self, object_name: str) -> dict | list | None:
        """
        Load JSON data from MinIO.
        
        Args:
            object_name: Name of the object to retrieve
            
        Returns:
            dict or list: Loaded Python object, or None if not found
        """
        try:
            # Download from MinIO
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            
            # Read and parse JSON
            data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            
            logger.info(f"Loaded data from MinIO: {object_name}")
            return data
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                logger.warning(f"Object not found in MinIO: {object_name}")
                return None
            else:
                logger.error(f"Error loading from MinIO: {str(e)}")
                raise e
    
    def delete_object(self, object_name: str) -> bool:
        """
        Delete object from MinIO.
        
        Args:
            object_name: Name of the object to delete
            
        Returns:
            bool: True if successful
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            logger.info(f"Deleted object from MinIO: {object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"Error deleting from MinIO: {str(e)}")
            raise e
    
    def list_objects(self, prefix: str = "") -> list[str]:
        """
        List objects in bucket with optional prefix.
        
        Args:
            prefix: Object name prefix to filter
            
        Returns:
            list: List of object names
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            object_names = [obj.object_name for obj in objects]
            logger.info(f"Found {len(object_names)} objects in MinIO with prefix '{prefix}'")
            return object_names
            
        except S3Error as e:
            logger.error(f"Error listing objects: {str(e)}")
            raise e


def get_minio_client() -> MinioClient:
    """
    Factory function to get MinIO client instance.
    
    Returns:
        MinioClient: Configured MinIO client
    """
    return MinioClient(
        endpoint="minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        secure=False
    )


def save_task_data(task_id: str, run_id: str, data: dict | list) -> str:
    """
    Save task data to MinIO with standardized naming.
    Args:
        task_id: Airflow task ID
        run_id: Airflow DAG run ID
        data: Data to save
        
    Returns:
        str: Object name used for storage
    """
    client = get_minio_client()
    object_name = f"{run_id}/{task_id}.json"
    client.save_json(object_name, data)
    return object_name


def load_task_data(task_id: str, run_id: str) -> dict | list | None:
    """
    Load task data from MinIO.

    Args:
        task_id: Airflow task ID
        run_id: Airflow DAG run ID
        
    Returns:
        dict or list: Loaded data, or None if not found
    """
    client = get_minio_client()
    object_name = f"{run_id}/{task_id}.json"
    return client.load_json(object_name)

