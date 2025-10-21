#!/usr/bin/env python3
"""
Script to run data ingestion manually.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import preprocess_data
from src.vector_store import add_documents_to_store
from src.utils import setup_logging

logger = setup_logging()

def main():
    """Main ingestion function."""
    try:
        logger.info("Starting data ingestion...")

        chunks = preprocess_data()
        add_documents_to_store(chunks)

        logger.info("Data ingestion completed!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
