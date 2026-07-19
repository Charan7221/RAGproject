"""
Helper Utilities Module

Contains utility functions for:
- Creating sample documents for testing
- Setting up logging configuration
- Data preprocessing helpers
"""

import os
import logging
from typing import Dict, Any

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Set up logging configuration for the RAG system with filename and line numbers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Enhanced log format with filename and line number
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Create handlers list
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Suppress some verbose library logs but keep our enhanced format
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

def create_sample_documents(output_dir: str = "data/sample_docs") -> str:
    """
    Create sample documents for testing the RAG system.
    
    Args:
        output_dir: Directory to save sample documents
        
    Returns:
        Path to the created documents directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample document 1: Healthcare Data Science
    healthcare_doc = """# Healthcare Data Science Applications

## Introduction
Healthcare data science involves the application of statistical and computational methods to healthcare data to improve patient outcomes, reduce costs, and enhance the quality of care.

## Key Applications

### Predictive Analytics
Machine learning models can predict patient readmissions, disease progression, and treatment outcomes. These models analyze historical patient data to identify patterns and risk factors.

### Drug Discovery
AI and machine learning accelerate drug discovery by analyzing molecular structures, predicting drug interactions, and identifying potential therapeutic compounds.

### Medical Imaging
Deep learning models, particularly convolutional neural networks (CNNs), are used for medical image analysis including X-ray interpretation, MRI analysis, and pathology detection.

### Electronic Health Records (EHR)
Natural language processing (NLP) techniques extract valuable insights from unstructured EHR data, enabling better clinical decision-making.

## Challenges
- Data privacy and security concerns
- Regulatory compliance (HIPAA, GDPR)
- Data quality and standardization issues
- Integration of multiple data sources

## Future Trends
The future of healthcare data science includes personalized medicine, real-time monitoring through IoT devices, and AI-powered diagnostic tools.
"""
    
    with open(os.path.join(output_dir, "healthcare_data_science.txt"), "w") as f:
        f.write(healthcare_doc)
    
    # Sample document 2: Financial Data Science
    finance_doc = """# Financial Data Science and Analytics

## Overview
Financial data science applies quantitative methods to financial data for risk management, algorithmic trading, fraud detection, and investment analysis.

## Core Applications

### Risk Management
Statistical models assess credit risk, market risk, and operational risk. Value at Risk (VaR) models and stress testing are commonly used techniques.

### Algorithmic Trading
Machine learning algorithms analyze market data to execute trades automatically. High-frequency trading systems process thousands of transactions per second.

### Fraud Detection
Anomaly detection algorithms identify suspicious transactions and potential fraud. These systems use supervised and unsupervised learning techniques.

### Credit Scoring
Predictive models assess the creditworthiness of loan applicants using historical data and alternative data sources.

## Technologies Used
- Python and R for statistical analysis
- Apache Spark for big data processing
- TensorFlow and PyTorch for deep learning
- Time series databases for market data storage

## Regulatory Considerations
Financial institutions must comply with regulations like Basel III, Dodd-Frank, and MiFID II when implementing data science solutions.

## Market Trends
The adoption of cloud computing, real-time analytics, and explainable AI is transforming the financial services industry.
"""
    
    with open(os.path.join(output_dir, "financial_data_science.txt"), "w") as f:
        f.write(finance_doc)
    
    # Sample document 3: Educational Technology
    education_doc = """# Educational Technology and Data Science

## Introduction
Educational data science leverages student data to improve learning outcomes, personalize education, and optimize educational processes.

## Key Areas

### Learning Analytics
Analysis of student behavior, performance, and engagement data to identify learning patterns and predict academic success.

### Adaptive Learning Systems
AI-powered platforms that adjust content difficulty and learning paths based on individual student performance and learning style.

### Student Performance Prediction
Machine learning models predict student dropout risk, course completion rates, and final grades using historical academic data.

### Curriculum Optimization
Data-driven approaches to curriculum design that identify the most effective learning sequences and content delivery methods.

## Implementation Challenges
- Student privacy protection
- Data integration from multiple systems
- Teacher training and adoption
- Ensuring educational equity

## Tools and Technologies
- Learning Management Systems (LMS)
- Educational data mining tools
- Statistical software (R, Python, SPSS)
- Visualization platforms (Tableau, Power BI)

## Future Directions
The integration of virtual reality, natural language processing for automated essay grading, and blockchain for credential verification represents the future of educational technology.
"""
    
    with open(os.path.join(output_dir, "educational_data_science.txt"), "w") as f:
        f.write(education_doc)
    
    logging.info(f"Created sample documents in {output_dir}")
    return output_dir

def format_qa_result(result: Dict[str, Any]) -> str:
    """
    Format QA result for display.
    
    Args:
        result: QA result dictionary
        
    Returns:
        Formatted string representation
    """
    formatted = []
    formatted.append(f"Question: {result['question']}")
    formatted.append(f"Answer: {result['answer']}")
    formatted.append(f"Sources ({len(result['source_documents'])}):")
    
    for i, doc in enumerate(result['source_documents'], 1):
        formatted.append(f"  {i}. {doc['source']} ({doc.get('type', 'unknown')})")
        formatted.append(f"     {doc['content'][:100]}...")
    
    return "\n".join(formatted)

def validate_system_requirements() -> Dict[str, bool]:
    """
    Validate that all required dependencies are available.
    
    Returns:
        Dictionary indicating which requirements are met
    """
    requirements = {}
    
    try:
        import torch
        requirements['pytorch'] = True
    except ImportError:
        requirements['pytorch'] = False
    
    try:
        import transformers
        requirements['transformers'] = True
    except ImportError:
        requirements['transformers'] = False
    
    try:
        import langchain
        requirements['langchain'] = True
    except ImportError:
        requirements['langchain'] = False
    
    try:
        import faiss
        requirements['faiss'] = True
    except ImportError:
        requirements['faiss'] = False
    
    try:
        import sentence_transformers
        requirements['sentence_transformers'] = True
    except ImportError:
        requirements['sentence_transformers'] = False
    
    return requirements 