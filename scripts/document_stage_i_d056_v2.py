"""Create amended protocol from immutable v1, documenting the exact replacement."""
from stage_i_d056_v2 import *

def main():
    text=(ROOT/'outputs/stage_i/d056_diagnostic_preparation_v1/REVISED_MINIMAL_GPU_PROTOCOL.md').read_text(encoding='utf-8')
    text=text.replace('# D-056 — Revised minimal diagnostic, prepared only','# D-056-v2 — Final amended minimal diagnostic, prepared only')
    text=text.replace('stage_i_d056_minimal_diagnostic_v1','stage_i_d056_minimal_diagnostic_v2')
    old='''To preserve144 forwards, cache probes are now **p=11 and12**, with three calls
per probe. Position3 remains in all teacher-forced/trace diagnostics; it is not
a matched-cache probe in this version. This is a disclosed refinement of an
unexecuted proposal, requiring approval before running, not a changed past result.'''
    new='''The reviewed v1 proposed p=11/12. Under the researcher-requested v2 amendment,
cache probes are **p=3 and11**, with three calls per probe. D055 observed wrong
marker selection at p3 and newline failure at p11; the new pair directly probes
these existing failures. p12 stays in teacher-forced row11, and visual-boundary
measurements remain in the same TF budget. No new forward or numerical tolerance
is introduced. The immutable v1 proposal is preserved as historical evidence.'''
    require(old in text,'Parent protocol passage changed');text=text.replace(old,new)
    text += '''

## V2 implementation-validation additions

See D056_V2_AMENDMENT.md and the machine-readable fingerprint/runtime binding
reports. Independent raw-top10 and processed-top10 rankings include exact IDs and
float32 values, with ascending-ID ties; processed masked -inf is explicit.
Selected beam ancestry directly gathers both raw logits and processed scores for
every emitted token, separately from library transition-score reconstruction.
The pinned vectorized finalization route does not append EOS. EOS at the selected
lineage and the global censor event are separate; no full768 winner is inferred.

Historical bare-Parameter tensor hashing is byte-compatible with D052/D054.
Actual adapter payloads and record hashes are checked without model loading;
backbone named-Parameter rehashing remains a pre-forward execution guard. The
collator Python body is exercised using a disclosed NumPy-only torch shim and
independent expected masks. Real Torch collation, actual forwarding and adapter
attachment remain pending. The future runner compares actual padded arrays and
beam-expanded source/global masks against these frozen offline bindings.
'''
    write(OUT/'REVISED_MINIMAL_GPU_PROTOCOL.md',text)
    write(OUT/'README_FIRST.md','''# D-056-v2 — Final offline preparation for review

Read D056_V2_AMENDMENT.md, FINAL_READINESS_REPORT.md and
PROPOSED_GPU_EXECUTION_AUTHORIZATION.md. Cache probes are p=3/11; 24/144/36
ceilings and all scientific/numerical settings remain fixed. No GPU authorization
has been granted. The template remains false. D051 strict replay FAIL is unchanged.

The accepted literature rationale and interpretation tree are preserved. No model
was loaded and no model/GPU/Teacher/test/training execution took place. This is
offline readiness, not evidence of GPU feasibility or correct generation.

The bundle verifier checks exported bytes/tokenization/alignment only. Full
workspace regression requires the original historical dataset/checkpoint/runtime
paths; the portable ZIP does not contain the pretrained backbone or full datasets.
''')
    write(OUT/'CHATGPT_REVIEW_REQUEST.md','''# Independent D-056-v2 preparation review

Check that p=3/11 replaces p=11/12 for the pre-observed failures, with p12 and
visual boundaries retained in teacher forcing and budgets unchanged. Verify
historical serializer evidence, independent raw/processed rankings, selected
beam ancestry and score reconstruction, EOS versus censor/finalization, exact
source/mask/adapter bindings, preserved v1 evidence and D051 FAIL, and unchanged
numerical screens. Review limitations of AST/NumPy tests versus real GPU execution.

Do not infer GPU authorization or an architecture winner. The researcher's next
decision is whether to authorize only the exact hash-bound scope. No training,
Teacher regeneration, annotation, test evaluation or dataset changes are proposed.
''')

if __name__=='__main__':main()
