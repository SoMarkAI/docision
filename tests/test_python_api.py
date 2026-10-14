import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from docision import Docision

Q=[{'question':'Table?','candidates':['yes','no']}]
class PythonAPITests(unittest.TestCase):
    def test_instances_are_independent_and_close_does_not_close_another(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.png';p.write_bytes(b'fixture')
            a,b=object(),object()
            with patch('docision.api.load_model',side_effect=[(a,('a','cpu'),{}),(b,('b','cpu'),{})]),patch('docision.api.score_file',return_value={'rows':[{'page':1}]}) as score:
                first=Docision('a',state_dir=d);second=Docision('b',state_dir=d)
                self.assertEqual(first.inspect(p,questions=Q),[{'page':1}]);self.assertIs(score.call_args.args[1],a)
                first.close()
                with self.assertRaises(RuntimeError):first.inspect(p,questions=Q)
                second.inspect(p,questions=Q);self.assertIs(score.call_args.args[1],b)
                second.close()
    def test_validation_precedes_inference_and_exports_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'empty.pdf';p.touch()
            with patch('docision.api.load_model',return_value=(object(),('a','cpu'),{})),patch('docision.api.score_file') as score:
                with Docision('a',state_dir=d) as model:
                    with self.assertRaises(ValueError):model.inspect(p,questions=Q)
                    with self.assertRaises(ValueError):model.inspect(p,questions=[{'question':'x','candidates':['same','same']}])
                    score.assert_not_called()
            rows=[{'file':'合同.pdf','page':1,'question_id':'q1','question':'签名？','candidates':['有','无'],'result':{'decision':{'text':'有'}}}]
            output=Docision.export(rows,Path(d)/'out.json');self.assertEqual(json.loads(output.read_text()),rows)
            with self.assertRaises(FileExistsError):Docision.export([],output)
            Docision.export(rows,Path(d)/'out.csv')
            self.assertIn('合同.pdf',(Path(d)/'out.csv').read_text())
if __name__=='__main__':unittest.main()
