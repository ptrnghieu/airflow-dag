import sys
import os

# Add dags folder to Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'dags'))

from arxiv_scraper import scrape_arxiv_papers, save_to_csv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scraper():
    """Test function to check scraper is working"""
    try:
        logger.info("Start testing ArXiv scraper...")
        
        # Test with simple query
        query = "machine learning"
        max_results = 5  # Only get 5 papers to test
        
        logger.info(f"Test with query: {query}, max_results: {max_results}")
        
        # Test scrape data
        papers = scrape_arxiv_papers(
            query=query,
            max_results=max_results,
            output_dir="./test_output"
        )
        
        if papers:
            logger.info(f"Scraped {len(papers)} papers")
            
            # Show first paper information
            if len(papers) > 0:
                first_paper = papers[0]
                logger.info("📄 First paper:")
                logger.info(f"  - Title: {first_paper['title'][:100]}...")
                logger.info(f"  - Authors: {first_paper['authors'][:100]}...")
                logger.info(f"  - Published: {first_paper['published']}")
                logger.info(f"  - Categories: {first_paper['categories']}")
            
            # Test save data
            logger.info("Test save data...")
            save_to_csv(output_dir="./test_output")
            logger.info("Saved data successfully")
            
        else:
            logger.warning("Couldn't scrape any papers")
            
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        raise e

def test_imports():
    """Test import necessary modules"""
    try:
        logger.info("Test import các modules...")
        
        import arxiv
        logger.info("arxiv module imported successfully")
        
        import pandas as pd
        logger.info("pandas module imported successfully")
        
        from datetime import datetime
        logger.info("datetime module imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Error importing module: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Start testing ArXiv Scraper")
    logger.info("=" * 50)
    
    # Test imports before
    if test_imports():
        # Test scraper
        test_scraper()
        logger.info("=" * 50)
        logger.info("Test completed! Scraper is working normally.")
        logger.info("Check ./test_output directory to see the result")
    else:
        logger.error("Test failed due to import modules error")
        sys.exit(1)

