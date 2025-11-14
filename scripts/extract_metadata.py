# scripts/extract_metadata.py
"""
One-time script to extract structured metadata using LLM
Run this once to enrich your dataset before indexing
Then use the output to index the data into the vector database (Chroma) for RAG retrieval
"""

"""
Output need to have at least these information:
- Uniq Id
- Product Name
- Selling Price
- Product Category
- Product Brand
- Product Material
- Product Description
- Product Review Score
"""
