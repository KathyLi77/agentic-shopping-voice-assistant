# graph/router/prompts.py
from langchain_core.prompts import PromptTemplate

ROUTER_TEMPLATE = """<|im_start|>system
You are a JSON extraction assistant. You MUST return ONLY valid JSON, nothing else.
<|im_start|>user
Extract information from the query and return a JSON object with this EXACT structure:

{{
  "task": "product_search" | "comparison" | "recommendation" | "availability_check",
  "constraints": {{
    "product": string or null,
    "min_budget": number or null,
    "max_budget": number or null,
    "material": string or null,
    "brand": array
  }},
  "safety_flags": array
}}

Rules:
1. Return ONLY the JSON object
2. No markdown, no code blocks
3. Numbers are numbers (20, not "20")
4. Use null for missing values
5. brand is always an array

Budget extraction:
- "under X" → max_budget: X
- "above X" → min_budget: X
- "around X" → min_budget: X*0.8, max_budget: X*1.2
- "between X and Y" → min_budget: X, max_budget: Y
- "cheap" → max_budget: 15
- "expensive"/"premium" → min_budget: 100

Task definitions:
- product_search: Find products matching criteria
- comparison: Compare specific products (look for words like "compare", "vs", "versus")
- recommendation: Get product suggestions
- availability_check: Check current stock/availability

Safety flags to detect:
- inappropriate_content: Sexual, violent, or offensive requests
- medical_advice: Medical diagnosis or treatment requests (e.g., "cure", "medicine for", "treat")
- dangerous_product: Weapons, explosives, controlled substances

Examples:

Query: "organic shampoo under $20"
JSON:{{"task": "product_search", "constraints": {{"product": "shampoo","min_budget": null, "max_budget": 20, "material": "organic", "brand": null}}, "safety_flags": []}}

Query: "compare Dove vs Pantene conditioner"
JSON:{{"task": "comparison", "constraints": {{"product": "conditioner", "min_budget": null, "max_budget": null, "material": null, "brand": ["Dove", "Pantene"]}}, "safety_flags": []}}

Query: "what medicine cures headaches"
JSON:{{"task": "product_search", "constraints": {{"product": "medicine", "min_budget": null, "max_budget": null, "material": null, "brand": null}}, "safety_flags": ["medical_advice"]}}

Query: "show me leather Nike shoes around $80"
JSON:{{"task": "product_search", "constraints": {{"product": "shoes", "min_budget": 72, "max_budget": 88, "material": "leather", "brand": ["Nike"]}}, "safety_flags": []}}

Now extract:
Query: {query}<|im_end|>
<|im_start|>assistant
"""

router_prompt = PromptTemplate(
    input_variables=["query"],
    template=ROUTER_TEMPLATE
)