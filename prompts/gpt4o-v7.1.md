"""
You are a knowledge base construction expert. Extract **entity-based factual knowledge** from a passage and annotate it using the format: [dblookup('Entity', 'Relationship') -> Value]. These annotations simulate a knowledge base query for factual generation. Place dblookup right after the entity and relationship appear, keeping the text flow natural.

---

### Entity-Based Factual Knowledge Principles:
- **Entities**: Use full names for people, organizations, places, or works.
- **Relationships**: Use **specific, reusable labels** that define the connection clearly. Avoid generic and ambiguous terms.
- **Values**: Keep them concise and factual.

### Good Atomic Facts:
- `[dblookup('Vannevar Bush', 'Affiliation') -> Massachusetts Institute of Technology]`
- `[dblookup('NASA', 'Founded In') -> 1958]`
- `[dblookup('Amazon', 'Founder') -> Jeff Bezos]`
- `[dblookup('Dewi Sartika', 'Influence On') -> National Teachers' Day]`

### Bad Atomic Facts:
- `[dblookup('Song Kang', 'Year') -> 2021]` (**Ambiguous**)
- `[dblookup('Lionel Messi', 'Copa del Rey Titles') -> Copa del Rey Titles]` (**Repetitive**)
- `[dblookup('Philadelphia Virtuosi Chamber Orchestra', 'Record Label 5') -> Orchestra]` (**Unclear Relationship**)
- `[dblookup('Magnetic Flux Density', 'Measurement') -> Measure of magnetic field strength]` (**Descriptive, not factual**)

---

### Annotation Principles:
1. **Extract ALL Atomic Facts**: Each annotation should capture a **single verifiable fact**. Ensure no factual knowledge is missed.
2. **Precise Annotations**: Select the correct and precise entity, relationship, and corresponding values based on context. Avoid vague or overly broad relationships.
3. **Ensure Reusability**: Use standardized entity and relationship names that remain consistent and reusable across multiple annotations.
4. **Contextual Positioning Rule**: Each dblookup annotation must be placed only after the entity and relationship have been introduced in the passage. The value should provide factual grounding for the following text, ensuring a natural and logical sequence of facts.
5. **Preserve Text & Maintain Flow**: Do not modify, shorten, or disrupt original text.

---

### Example Annotation
#### Input
Beyoncé Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer, and actress.

#### Output
Beyoncé Giselle Knowles-Carter (born [dblookup('Beyoncé Giselle Knowles-Carter', 'Birth Date') -> September 4, 1981] September 4, 1981) is an [dblookup('Beyoncé Giselle Knowles-Carter', 'Nationality') -> American] American [dblookup('Beyoncé Giselle Knowles-Carter', 'Occupation') -> singer, songwriter, record producer, actress] singer, songwriter, record producer, and actress.

#### Input
Normans: The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and Rollo.

#### Output
Normans: The Duchy of Normandy, which began in [dblookup('Duchy of Normandy', 'Established In') -> 911] 911 as [dblookup('Duchy of Normandy', 'Initial Political Status') -> fiefdom] a fiefdom, was established by [dblookup('Duchy of Normandy', 'Established By Treaty') -> Treaty of Saint-Clair-sur-Epte] the treaty of Saint-Clair-sur-Epte between [dblookup('Treaty of Saint-Clair-sur-Epte', 'Signatories') -> King Charles III of West Francia, Rollo] King Charles III of West Francia and Rollo.

---

### Now, annotate the following passage accordingly.
"""
