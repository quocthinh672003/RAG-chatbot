"""
Test script for Haystack pipelines
"""

import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_minimal_haystack():
    """Test minimal haystack pipeline"""
    try:
        from services.haystack_minimal_pipeline import get_minimal_haystack_pipeline
        
        logger.info("🧪 Testing Minimal Haystack Pipeline...")
        
        # Get pipeline
        pipeline = get_minimal_haystack_pipeline()
        
        # Test documents
        test_docs = [
            {
                "page_content": "Việt Nam là một quốc gia ở Đông Nam Á với dân số khoảng 97 triệu người.",
                "metadata": {
                    "source_name": "test_doc_1",
                    "page": 1,
                    "file_type": "txt",
                    "language": "vi"
                }
            },
            {
                "page_content": "Hà Nội là thủ đô của Việt Nam và là trung tâm chính trị, văn hóa của cả nước.",
                "metadata": {
                    "source_name": "test_doc_2", 
                    "page": 1,
                    "file_type": "txt",
                    "language": "vi"
                }
            }
        ]
        
        # Add documents
        pipeline.add_documents(test_docs)
        
        # Test query
        result = pipeline.query("Việt Nam")
        
        logger.info(f"✅ Minimal Haystack Test Result: {result['answer'][:100]}...")
        logger.info(f"📊 Document count: {pipeline.get_document_count()}")
        logger.info(f"📋 Pipeline info: {pipeline.get_pipeline_info()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Minimal Haystack Test Failed: {e}")
        return False

def test_simple_haystack():
    """Test simple haystack pipeline"""
    try:
        from services.haystack_simple_pipeline import get_simple_haystack_pipeline
        
        logger.info("🧪 Testing Simple Haystack Pipeline...")
        
        # Get pipeline
        pipeline = get_simple_haystack_pipeline()
        
        # Test documents
        test_docs = [
            {
                "page_content": "OpenAI là công ty nghiên cứu AI hàng đầu thế giới.",
                "metadata": {
                    "source_name": "test_doc_3",
                    "page": 1,
                    "file_type": "txt",
                    "language": "vi"
                }
            }
        ]
        
        # Add documents
        pipeline.add_documents(test_docs)
        
        # Test query
        result = pipeline.query("OpenAI")
        
        logger.info(f"✅ Simple Haystack Test Result: {result['answer'][:100]}...")
        logger.info(f"📊 Document count: {pipeline.get_document_count()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple Haystack Test Failed: {e}")
        return False

def test_haystack_only():
    """Test haystack-only pipeline"""
    try:
        from services.haystack_only_pipeline import get_haystack_only_pipeline
        
        logger.info("🧪 Testing Haystack-Only Pipeline...")
        
        # Get pipeline
        pipeline = get_haystack_only_pipeline()
        
        # Test documents
        test_docs = [
            {
                "page_content": "RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp retrieval và generation.",
                "metadata": {
                    "source_name": "test_doc_4",
                    "page": 1,
                    "file_type": "txt",
                    "language": "vi"
                }
            }
        ]
        
        # Add documents
        pipeline.add_documents(test_docs)
        
        # Test query
        result = pipeline.query("RAG")
        
        logger.info(f"✅ Haystack-Only Test Result: {result['answer'][:100]}...")
        logger.info(f"📊 Document count: {pipeline.get_document_count()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Haystack-Only Test Failed: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline"""
    try:
        from services.rag_pipeline import rag_pipeline
        
        if rag_pipeline is None:
            logger.warning("⚠️ RAG pipeline not available")
            return False
        
        logger.info("🧪 Testing RAG Pipeline...")
        
        # Test documents
        test_docs = [
            {
                "page_content": "Haystack là framework RAG mạnh mẽ cho Python.",
                "metadata": {
                    "source_name": "test_doc_5",
                    "page": 1,
                    "file_type": "txt",
                    "language": "vi"
                }
            }
        ]
        
        # Add documents
        rag_pipeline.add_documents(test_docs)
        
        # Test query
        result = rag_pipeline.query("Haystack")
        
        logger.info(f"✅ RAG Pipeline Test Result: {result['answer'][:100]}...")
        logger.info(f"📊 Document count: {rag_pipeline.get_document_count()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG Pipeline Test Failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Haystack Pipeline Tests...")
    
    results = []
    
    # Test minimal haystack
    results.append(("Minimal Haystack", test_minimal_haystack()))
    
    # Test simple haystack
    results.append(("Simple Haystack", test_simple_haystack()))
    
    # Test haystack-only
    results.append(("Haystack-Only", test_haystack_only()))
    
    # Test RAG pipeline
    results.append(("RAG Pipeline", test_rag_pipeline()))
    
    # Print summary
    logger.info("\n📋 Test Summary:")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Haystack pipelines are working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
