"""Offline string/provenance regression tests; no ML imports or inference."""
import unittest
import source_bound_visual_diagnostic_v1 as d


def case(caps):
    source='name=Synthetic café\n[REV_1] review\n'+''.join(
        '['+key+'] '+value+'\n' for key,value in caps.items())
    return dict(split='train',condition='filtered',source=source,
                source_sha256=d.sha_bytes(source.encode('utf-8')),captions=caps)


class DiagnosticTests(unittest.TestCase):
    def test_review_bytes_and_original_CRLF_preserved(self):
        original='[REVIEW_CONSENSUS]\r\nCafé — mixed opinions.\r\n\r\n[VISUAL_EVIDENCE]\r\nOld.\r\n'
        alt,trace,checks=d.render(case({'IMG_CAP_1':'a cup.'}),original)
        self.assertEqual(d.section_parts(original)[0].encode(),d.section_parts(alt)[0].encode())
        self.assertTrue(checks['prefix_bytes_exact'])
        self.assertIn('[VISUAL_EVIDENCE]\r\nCaption 1: a cup.\n',alt)

    def test_zero_captions_leave_empty_visual(self):
        original='[REVIEW_CONSENSUS]\nReview.\n[VISUAL_EVIDENCE]\nInvented visual.'
        alt,trace,checks=d.render(case({}),original)
        self.assertEqual(d.section_parts(alt)[2],'')
        self.assertEqual(trace,[])
        self.assertTrue(checks['alternative_visual_body_empty'])

    def test_all_captions_once_in_original_order_including_duplicate_text(self):
        caps={'IMG_CAP_1':'a cup.','IMG_CAP_2':'a cup.','IMG_CAP_3':'a plate.'}
        original='[REVIEW_CONSENSUS]\nReview.\n[VISUAL_EVIDENCE]\nOld.'
        alt,trace,checks=d.render(case(caps),original)
        self.assertEqual(d.section_parts(alt)[2],
                         'Caption 1: a cup.\nCaption 2: a cup.\nCaption 3: a plate.\n')
        self.assertEqual(len(trace),3)

    def test_unicode_character_and_UTF8_offsets_recover_exact_text(self):
        c=case({'IMG_CAP_1':'a café — café.'})
        original='[REVIEW_CONSENSUS]\nRésumé.\n[VISUAL_EVIDENCE]\nOld.'
        alt,trace,_=d.render(c,original)
        t=trace[0]
        self.assertEqual(c['source'].encode()[t['source_UTF8_byte_start']:t['source_UTF8_byte_end']].decode(),t['text'])
        self.assertEqual(alt.encode()[t['alternative_UTF8_byte_start']:t['alternative_UTF8_byte_end']].decode(),t['text'])

    def test_wrong_source_hash_rejected(self):
        c=case({'IMG_CAP_1':'a cup.'});c['source']+='changed'
        with self.assertRaisesRegex(ValueError,'Source hash'):d.exact_captions(c)

    def test_caption_order_or_rewritten_caption_rejected(self):
        c=case({'IMG_CAP_1':'a cup.','IMG_CAP_2':'a plate.'})
        c['captions']=dict(reversed(list(c['captions'].items())))
        with self.assertRaisesRegex(ValueError,'order/identity'):d.exact_captions(c)
        c=case({'IMG_CAP_1':'a cup.'});c['captions']['IMG_CAP_1']='a glass.'
        with self.assertRaisesRegex(ValueError,'body differs'):d.exact_captions(c)

    def test_nontrain_or_nonfiltered_rejected(self):
        c=case({});c['split']='validation'
        with self.assertRaisesRegex(ValueError,'TRAIN Filtered'):d.exact_captions(c)
        c=case({});c['condition']='unfiltered'
        with self.assertRaisesRegex(ValueError,'TRAIN Filtered'):d.exact_captions(c)

    def test_ambiguous_section_headers_rejected(self):
        c=case({})
        for original in ('no headers', '[VISUAL_EVIDENCE]\n[REVIEW_CONSENSUS]\nx',
                         '[REVIEW_CONSENSUS]\nx\n[VISUAL_EVIDENCE]\nx\n[VISUAL_EVIDENCE]\n'):
            with self.assertRaises(ValueError):d.render(c,original)


if __name__=='__main__':unittest.main(verbosity=2)
