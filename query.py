"""
query.py - Enhanced query using specialized Haystack retrievers
"""
from utils.retrievers import retrieve_documents
from config import LLM_MODEL, OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def retrieve_and_answer(query_text: str, top_k: int = 10):
    """
    Enhanced RAG pipeline with specialized retrievers:
    1. Detect file types in query (optional)
    2. Use specialized retriever for better results
    3. Apply advanced ranking pipeline
    4. Generate answer with LLM
    """
    
    # 1. Retrieve documents using specialized pipeline
    docs = retrieve_documents(query_text, file_type=None, top_k=top_k)
    
    if not docs:
        return {"answer": "Không tìm thấy tài liệu liên quan.", "sources": [], "total_sources": 0}
    
    # 2. Generate answer using LLM
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1,
    )
    
    # Create context from documents
    context = "\n\n".join([doc.content for doc in docs])
    
    # Create enhanced prompt with file type awareness
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Bạn là một trợ lý AI thông minh. Dựa trên các tài liệu sau đây, hãy trả lời câu hỏi một cách chính xác và đầy đủ.
        
        **Hướng dẫn:**
        - Chỉ trả lời dựa trên thông tin có trong tài liệu được cung cấp
        - Nếu không thể trả lời từ tài liệu, hãy nói rõ "Tôi không có đủ thông tin để trả lời câu hỏi này"
        - Trả lời bằng tiếng Việt, mạch lạc và có cấu trúc
        - Nếu có nhiều thông tin liên quan, hãy tổ chức thành các điểm chính
        - Chú ý đến loại tài liệu (PDF, Excel, JSON, v.v.) để trả lời phù hợp
        
        **Tài liệu tham khảo:**
        {context}
        
        **Câu hỏi:** {question}
        
        **Trả lời:**"""
    )
    
    # Generate answer
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(context=context, question=query_text)
    
    # Prepare sources with enhanced metadata
    sources = []
    for i, doc in enumerate(docs, 1):
        source_info = {
            "rank": i,
            "content": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
            "source": doc.meta.get("source_name", "Unknown"),
            "page": doc.meta.get("page_number", "Unknown"),
            "score": doc.score if hasattr(doc, 'score') else None,
            "file_type": doc.meta.get("file_type", "Unknown"),~
            "file_size": doc.meta.get("file_size", "Unknown"),
            "processor_type": doc.meta.get("processor_type", "Unknown"),
            "chunk_size": doc.meta.get("chunk_size", "Unknown"),
            "chunk_overlap": doc.meta.get("chunk_overlap", "Unknown")
        }
        sources.append(source_info)
    
    return {
        "answer": result.strip(),
        "sources": sources,
        "total_sources": len(sources)
    }