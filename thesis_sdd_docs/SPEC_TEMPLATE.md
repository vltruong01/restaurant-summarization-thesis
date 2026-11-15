# Stage Specification — <stage name>

**Spec ID:** XXX  
**Stage:**  
**Status:** Draft / Approved / Implemented / Validated  
**Last updated:**  
**Execution environment:** Local / Vast.ai / Both / Human annotation / API  
**Environment gate required:** A1 / A2 / Other  

## 1. Objective
What this stage must accomplish.

## 2. Scope
### In scope
- 

### Out of scope
- 

## 3. Inputs
- file/dataset:
- required fields:
- upstream dependency:
- immutable manifest/checksum dependencies:
- data root and portable path convention:

## 4. Outputs / artifacts
- 

## 5. Methodology rules and invariants
Rules that must not change during implementation.
- 

## 6. Configuration
### Fixed values
- 
- execution environment/dependency lock:
- tokenizer/model revision when applicable:

### TBD / empirical values
- 

For every empirical value, state:
- candidate values/search space,
- validation data allowed,
- selection metric/rule,
- forbidden data (especially held-out test).

## 7. Implementation tasks
1. 
2. 

## 8. Validation / sanity checks
- 
- If moving environments: verify artifact hashes, split/review manifests, and tokenizer behavior before processing.
- If using rented compute: validate resume state and durable artifact synchronization.

## 9. Acceptance criteria
A stage can be marked complete only if every required criterion passes.
- [ ] 
- [ ] 

## 10. Stop conditions
Stop and ask the user if:
- 

## 11. Expected logs / reports
- `THESIS_PROGRESS.md` entry
- `DECISIONS.md` entry if needed
- 

## 12. Deviations / amendments
Any approved change to this spec must be recorded here and in `DECISIONS.md` before implementation continues.
