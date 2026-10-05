import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_benchmark.artifacts import materialize, read_json, RESOURCES
from decision_benchmark.cli import compare_points, run_model, verify_resources
from systematic.confirmation_analysis import validate_run, ValidationError

class ReleasedArtifactsTests(unittest.TestCase):
    def test_resource_fingerprints(self):
        self.assertGreater(verify_resources()['files_verified'],40)

    def test_text_free_run_validates_every_recorded_request(self):
        with tempfile.TemporaryDirectory() as directory:
            run=materialize('core',Path(directory)/'run','jev')
            manifest,cases,requests,predictions,index=validate_run(run)
            self.assertEqual(manifest['case_representation'],'text-index-v1')
            self.assertEqual(len(cases),1456)
            self.assertEqual(len(requests),39168)
            self.assertEqual(set(requests),set(predictions))
            self.assertTrue(all('text' not in case for case in cases.values()))

    def test_changed_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run=materialize('core',Path(directory)/'run','jev',quick=True)
            with (run/'requests.jsonl').open('ab') as stream:stream.write(b'\n')
            with self.assertRaisesRegex(ValidationError,'Changed frozen file'):
                validate_run(run)

    def test_duplicate_text_identity_is_rejected_even_without_text(self):
        with tempfile.TemporaryDirectory() as directory:
            run=materialize('core',Path(directory)/'run','jev',quick=True)
            cases=[json.loads(line) for line in (run/'cases.jsonl').read_text().splitlines()]
            cases[1]['normalized_text_sha256']=cases[0]['normalized_text_sha256']
            content=''.join(json.dumps(case)+'\n' for case in cases).encode()
            (run/'cases.jsonl').write_bytes(content)
            manifest=read_json(run/'manifest.json')
            manifest['file_sha256']['cases.jsonl']=hashlib.sha256(content).hexdigest()
            (run/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValidationError,'Duplicate text'):
                validate_run(run)

    def test_prepare_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'Refusing to overwrite'):
                materialize('core',directory)

    def test_inference_cannot_consume_text_indices(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            run=materialize('core',Path(directory)/'run')
            with patch('systematic.jev_api.Client') as client:
                with self.assertRaisesRegex(ValueError,'requires text'):
                    run_model(SimpleNamespace(inputs=run))
                client.assert_not_called()

    def test_point_comparator_detects_a_rate_regression(self):
        with self.assertRaisesRegex(ValueError,'Point estimates differ'):
            compare_points({'rate':.4,'bootstrap95':[.2,.6]}, {'rate':.5,'bootstrap95':[.1,.9]})
        self.assertEqual(compare_points({'rate':.5,'bootstrap95':[.2,.6]}, {'rate':.5,'bootstrap95':[.1,.9]})['status'],'passed')

if __name__=='__main__':unittest.main()
