#!/usr/bin/env python3
"""
Script to setup the vector database by ingesting documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import preprocess_data
from src.vector_store import add_documents_to_store
from src.utils import validate_env_vars, setup_logging

logger = setup_logging()

def main():
    """Main setup function."""
    try:
        logger.info("Validating environment variables...")
        validate_env_vars()

        logger.info("Preprocessing data...")
        chunks = preprocess_data()

        logger.info("Adding documents to vector store...")
        add_documents_to_store(chunks)

        logger.info("Setup completed successfully!")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
