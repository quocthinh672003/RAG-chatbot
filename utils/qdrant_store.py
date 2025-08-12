"""
generate haystack qdrantDocumentStore và EmbeddingRetriever
"""

from haystack.document_stores import QdrantDocumentStore
from haystack.nodes import EmbeddingRetriever, DensePassageRetriever
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, QDRANT_EMBEDDING_DIMENSION, QDRANT_EMBEDDING_MODEL
# haystack.document_stores dùng để lưu trữ tài liệu trong Qdrant 
# haystack.nodes dùng để tạo retriever cho việc truy vấn tài liệu
#embedding retriever là chuyển đổi tài liệu từ văn bản sang vector để tìm kiếm tương tự

def get_qdrant_document_store(recreate: bool = False):
    """
    Generate or connect to a QdrantDocumentStore.
    """
    ds = QdrantDocumentStore(
        host = QDRANT_HOST,
        port = QDRANT_PORT,
        collection_name = QDRANT_COLLECTION_NAME, 
        prefer_grpc = False, # prefer_grpc là tùy chọn để sử dụng giao thức gRPC thay vì REST
        #gRPC là một giao thức gọi hàm từ xa, cho phép các ứng dụng giao tiếp với nhau qua mạng
        embedding_dim = QDRANT_EMBEDDING_DIMENSION, # embedding_dim là kích thước của vector embedding
        #vector embedding là một biểu diễn số học của văn bản 
        similarity = "cosinne", # cosine similarity là phương pháp đo lường độ tương đồng giữa các vector
        index = "hnsw" # hnsw là thuật toán tìm kiếm gần nhất trong không gian vector
    )
    if recreate:
        try:
            ds.delete_index(QDRANT_COLLECTION_NAME)
        except Exception:
            pass
    return ds
    
def get_embedding_retriever(document_store):
    """
    Create embedding retriever using sentence-transformers đồng bộ
    sentence-transformers là library để change text to vector embeddings
    """
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=QDRANT_EMBEDDING_MODEL,
        model_format="sentence_transformers",  # model_format là định dạng của mô hình embedding
        use_gpu=False,  # use_gpu là tùy chọn để sử dụng GPU nếu có
        scale_score=True  # scale_score là tùy chọn để chuẩn hóa điểm số truy vấn
    )
    return retriever
    