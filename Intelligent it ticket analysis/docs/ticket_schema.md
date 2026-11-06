
# IT-Sentinel - Unified Ticket Schema

## Objective
Create a common structure for tickets collected from multiple
datasets without losing original information or provenance.

| Field | Data Type | Description |
|---|---|---|
| ticket_id | String | Unique internal ticket identifier |
| source_record_id | String | Original dataset ticket ID, if available |
| source_dataset | String | Name of original dataset |
| record_origin | String | Real, de-identified or candidate-created |
| source_split | String | Train, validation, test or unknown |
| title | String | Ticket title |
| description | String | Original ticket description |
| symptoms | String | Identified symptoms |
| application | String | Affected application |
| service | String | Affected service |
| category | String | Ticket category |
| subcategory | String | Ticket subcategory |
| priority | String | P1, P2, P3 or P4 |
| intent | String | Incident, service request or other intent |
| impact | String | Business impact |
| urgency | String | Time sensitivity |
| status | String | Open, closed or resolved |
| assignment_group | String | Responsible team |
| environment | String | Production, development or testing |
| created_at | Datetime | Ticket creation timestamp |
| resolution_text | String | Historical resolution if available |

## Data Integrity Rules

1. Preserve original source dataset identifiers.
2. Never invent missing ticket descriptions or resolutions.
3. Keep missing values as null.
4. Clearly distinguish real records from candidate-created records.
5. Preserve original classification labels.
6. Do not merge incompatible classification taxonomies.
7. Maintain the original train/test split.
8. Keep raw datasets unchanged.

## Dataset Mapping

### Dataset A: IT Support Tickets
- description: text
- category: label_name
- remaining fields: null unless available

### Dataset B: Mendeley Help Desk
- preserve all available metadata
- map fields only after inspecting the actual CSV columns
- use messages for historical ticket retrieval

### Dataset C: Candidate Knowledge Base
- separate from historical tickets
- maintain article ID, source, version and approval status
