"""
Your task is to extract and annotate entity-based factual knowledge from the provided text. Identify and annotate specific entities, relationships, and values using the `dblookup` format: [dblookup('Entity', 'Relationship') -> Value]

### Annotation Guidelines  
- Inline Insertion: Insert dblookup before factual statements without altering the text.  
- Atomic Facts: Each dblookup should capture one verifiable fact.  
- Entities: Use full names for people, organizations, places, or works.  
- Relationships: Use specific, reusable labels (avoid vague terms).  
- Values: Keep them concise and factual.
"""