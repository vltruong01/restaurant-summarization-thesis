"""Verify non-applied overlap-policy impact and gallery references independently."""
from collections import defaultdict
import json
import pandas as pd,yaml
from photo_integrity_audit import ROOT,CONFIG,digest_file


def main():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory']
    validation=json.loads((out/'validation_report.json').read_text());assert validation['validation_passed']
    photos=pd.read_csv(out/'image_manifest.csv',keep_default_na=False);valid=photos[photos.readable]
    flags=pd.read_csv(out/'hypothetical_overlap_flags.csv',keep_default_na=False).set_index('photo_index')
    impact=pd.read_csv(out/'hypothetical_overlap_business_impact.csv',keep_default_na=False).set_index('business_id')
    rank={'train':0,'validation':1,'test':2};owners=defaultdict(int)
    for row in valid.itertuples():owners[row.pixel_sha256]=max(owners[row.pixel_sha256],rank[row.split])
    expected={r.photo_index:rank[r.split]<owners[r.pixel_sha256] for r in valid.itertuples()}
    assert set(expected)==set(flags.index)
    for index,value in expected.items():assert flags.loc[index,'would_exclude_for_hypothetical_overlap']==value,index
    for bid,g in valid.groupby('business_id'):
        kept=g[g.photo_index.map(lambda i:not expected[i])];row=impact.loc[bid]
        assert row.current_readable==len(g) and row.hypothetical_removed==len(g)-len(kept) and row.hypothetical_remaining==len(kept)
        assert row.hypothetical_remaining_unique_pixels==kept.pixel_sha256.nunique()
        assert row.hypothetical_below5==(len(kept)<5) and row.hypothetical_zero==(len(kept)==0) and not row.policy_applied
    lookup=photos.set_index('photo_index');examples=pd.read_csv(out/'visual_pair_examples.csv',keep_default_na=False)
    for row in examples.itertuples():
        a,b=lookup.loc[row.photo_index_a],lookup.loc[row.photo_index_b]
        assert a.split in ['train','validation'] and b.split in ['train','validation']
        for side,record in [('a',a),('b',b)]:assert getattr(row,'photo_id_'+side)==record.photo_id and getattr(row,'path_'+side)==record.relative_path
        assert row.raw_equal==(a.raw_sha256==b.raw_sha256) and row.pixel_equal==(a.pixel_sha256==b.pixel_sha256)
        assert row.hamming_distance==(int(a.phash64_hex,16)^int(b.phash64_hex,16)).bit_count()
    findings=json.loads((out/'photo_audit_findings.json').read_text());proposal=findings['hypothetical_overlap_policy_NOT_APPLIED']
    assert proposal['removed_photos']==sum(expected.values())==785
    assert proposal['businesses_below5']==impact.hypothetical_below5.sum()==124 and proposal['businesses_zero']==impact.hypothetical_zero.sum()==14
    assert proposal['affected_businesses']==impact.hypothetical_removed.gt(0).sum()==172 and not proposal['applied']
    global_groups=pd.read_csv(out/'global_duplicate_collision_groups.csv',keep_default_na=False)
    global_members=pd.read_csv(out/'global_group_memberships.csv',keep_default_na=False)
    overlapping_groups=set(global_groups.loc[global_groups.metric.eq('pixel')&global_groups.cross_split&global_groups.splits.str.contains('test'),'group_id'])
    candidates=set(global_members.loc[global_members.group_id.isin(overlapping_groups),'photo_index'])
    test=valid[valid.split.eq('test')];overlap_indices=set(test.photo_index)&candidates
    overlap_bids=set(test.loc[test.photo_index.isin(overlap_indices),'business_id'])
    clean_bids=set(test.business_id)-overlap_bids
    sensitivity=pd.read_csv(out/'test_image_disjoint_diagnostic_businesses.csv',keep_default_na=False).set_index('business_id')
    sensitivity_summary=json.loads((out/'test_image_disjoint_diagnostic_summary.json').read_text())
    assert set(sensitivity.index)==set(test.business_id)
    for bid,g in test.groupby('business_id'):
        n=sum(i in overlap_indices for i in g.photo_index);row=sensitivity.loc[bid]
        assert row.test_photos==len(g) and row.overlap_photos==n and row.image_disjoint_from_train_validation==(n==0) and row.all_test_photos_overlap==(n==len(g))
    assert len(overlap_indices)==sensitivity_summary['overlapping_test_photo_IDs']==123
    assert len(clean_bids)==sensitivity_summary['test_businesses_with_no_exact_overlap']==83
    assert len(overlap_bids)==sensitivity_summary['test_businesses_with_any_exact_overlap']==17
    assert int(sensitivity.all_test_photos_overlap.sum())==sensitivity_summary['test_businesses_all_photos_overlap']==8
    report={'status':'PASS','hypothetical_flag_rows_verified':len(expected),'business_counterfactuals_verified':len(impact),'visual_pair_references_verified':len(examples),'test_exact_overlap_diagnostic_businesses_verified':100,'image_disjoint_diagnostic_businesses':83,'evaluation_protocol_adopted':False,'policy_applied':False,'photos_deleted':False,'businesses_excluded':False,'verifier_sha256':digest_file(__file__)}
    (out/'overlap_proposal_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
