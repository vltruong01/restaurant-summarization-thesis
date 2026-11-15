"""Future official scorer wrapper. Human approvals remain mandatory; not run in Phase1."""
from pathlib import Path
import sys,argparse
from s1000_final_eval_core_v1 import *
def main():
    # Resolve shared arguments without consuming the frozen scorer's own CLI flags.
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--bert-model',type=Path,required=True)
    a,_=p.parse_known_args()
    wheel=a.preparation/'METRIC_ASSETS/bert_score-0.3.13-py3-none-any.whl'
    require(file_hash(wheel)=='bbbb4c7fcdaa46d7681aff49f37f96faa09ed74e1b150e659bdc6b58a66989b9','Wrong BERTScore implementation')
    sys.path.insert(0,str(wheel))
    from stage_k_bert_loader_compat_v1 import legacy_binary_cpu_load
    from score_s1000_automatic_prepared_v1 import main as score
    with legacy_binary_cpu_load(a.bert_model/'pytorch_model.bin'):score()
if __name__=='__main__':main()
