# C2 LED source/token-allocation audit (003l)

2026-10-06. User authorizes the proposed next step: add an LED input/token-allocation audit to the maintained notebook. This is measurement and exploratory allocation comparison, not authorization to freeze a final template, selector, source length or caption budget.

## Fixed sources and scope

- Use accepted v4.2 refined selections for all2,000 train/200 validation businesses, at existing review-block budgets3500/4000/4500. No new selection/search; identical whole review IDs/text/stars/dates/order/count and primary diagnostics. No test100 tuning or source examples from test.
- Read selected originals from the immutable English pool and business records from v1. Stream sources once; retain only the union of selected originals. Use the pinned cached LED tokenizer with truncation/padding disabled, no model/network/GPU/API.
- Protect input/config/tokenizer/frozen formatter hashes, accepted completion/validation/seal references and research_pending. Final research fields remain null. Do not copy datasets or store full inputs for allbusinesses; compact token ledger plus limited examples only. Keep new scientific output under25MB, excluding the previous-state snapshot.

## Declared provisional source templates

Compare three AUDIT profiles: no-business-metadata control, business name+categories, and name+categories+city+state. Values are exact strings from original business records; None/missing is empty string. No aggregate stars/review counts, attributes, coordinates, addresses, IDs or helpful-vote metadata introduced. Metadata field choice and outer section markers are provisional, not adopted training design. No extra task/teacher instruction is silently added; any final source-template change requires full re-encoding.

Preserve the accepted review formatter exactly: `[REV_i] stars=... date=...\n<complete original>` separated by two newlines, chronological dates/IDs. A proposed outer `[REVIEWS]\n` follows the optional `[BUSINESS]` section. The proposed multimodal structure appends `\n\n[CAPTIONS]\n`; its empty-header overhead can be measured, but it contains NO generated captions. Text-only has no caption section. Actual unfiltered/filtered/shuffled captions must later use the same shared review evidence and caption allocation under their approved formatting.

Encode the COMPLETE source once with special tokens. Verify two added special tokens through the encoder special-token mask; literal special-looking strings in original reviews are not removed or confused with added tokens. The old review-block lengths already include review markers/stars/date/separators and BOS/EOS; never add their special tokens again.

Report standalone review BODY sum, exact marked review-block tokens, exact metadata standalone tokens (no special), exact text-only base tokens, and the exact marginal base-minus-review delta. The BODY difference is formatting+special+joint-BPE accounting, not an independent additive header length. Full-source encoding is authoritative; component sums may differ at boundaries. Exact empty-caption-header source and marginal overhead are measured separately.

## Exploratory caption allowances and capacity

Use5000 as the diagnostic source ceiling from the existing approximate planned target, not a finalized training parameter. Compare allowances0/250/500/750/1000/1250/1500 tokens. Zero is the text-only control; positive allowances reserve the ENTIRE incremental caption block relative to text-only base, including caption section/line markers/IDs/separators/BPE boundary effects. They are not per-photo counts or measured captions. No fake caption text, BLIP-2 inference, k choice, length limit or semantic quality claim.

Exact available caption capacity =5000 - exact text-only base tokens. A hypothetical allowance fits iff it does not exceed this capacity; base+allowance is an allocation scenario, NOT actual caption-conditioned source length. Also show capacity after the measured empty header. Actual caption-conditioned length remains null until real captions are assembled and tokenized once.

Separate observed-set fits from the conservative declared-review-budget envelope using each business's measured marginal overhead: review_budget+overhead+allowance <=5000. The envelope covers only audited subsets/metadata, not arbitrary future inputs or unseen test metadata. Do not infer a fixed caption count from token capacity or guarantee all2300 inputs from train/validation measurement. Model padding/global attention/decoder targets/VRAM are outside this source-token audit.

## Verification and reporting

- Reconstruct/re-encode all6600 accepted complete review blocks, all19800 profile/base rows and empty-caption-header variants; compare against immutable IDs/count/tokens/primary diagnostics. Preserve input order, original text and flags. Independent validator reconstructs templates from the spec, checks exact base/body/marker/special/boundary accounting, IDs/text hashes, all63 allowance summaries and group summaries from originals; no accepted failure.
- Include count/below10 flags (no minimum required), exact token min/median/p95/max, metadata/marginal overhead, min/max/quantile capacity, source-only overflows, observed/conservative allowance fits, detailed extrema and the four existing primary count-drop cases. No review is removed/replaced/truncated for an audit allowance.
- Meaningful serialization edge checks: whole reviews with literal markers/Unicode, added-special-token accounting, unusually long metadata causing flagged overflow without truncation. Limited complete-source examples include metadata extremes, capacity extremes, low-count/source-short cases and allfour primary count drops. Not a content/human evaluation.
- Snapshot the accepted17-cell notebook/mutable handoff/docs/seals, add section9.6 to notebook00, execute cached/read-only, inspect new scientific plot and verify previous8 figures unchanged. Update progress/README/D-020/master plan with actual results/limits. No new notebook or bulk candidate cache.

Stop after the authorized audit and report concrete allocation options/tradeoffs. Final template, selector/allocations/2300 manifest remain pending; real caption pilot/image integrity and A2 feasibility remain later. If changing the planned source target, review allocation or research rule is required, record the proposal as TBD and wait for explicit approval rather than changing it automatically.

## Measured acceptance - 2026-10-06

Production139.111s/validator146.192s;2200/6600 original marked blocks/19800 source rows,103047 selected BODY originals/41804 full strings and441 summary rows/nine complete-source examples PASS. Two meaningful serializer edges PASS0.183s. Largest-profile overhead min34/median51/p9570/max115, source maxima3579/4083/4586 and minimum entire-caption room1421/917/414.4000+750 fits all2200 observed/envelope cases (max4833/4865), a concrete pilot option not adopted.4500+750 fits1160 and flags1040;4000+1000 flags736. No review changed, no actual captions, final research fields null. Notebook00 section9.6,18/18 cells/nine figures PASS17.847s; new plot inspected/previous8 unchanged, protected raw/cohort/English/splits unchanged. Prior17-cell handoff/notebook/cleanup seal snapshot retained; scientific outputs~18.3MB. Final template/selector/allocations/2300 manifest and image/real-caption/A2 feasibility remain pending.
