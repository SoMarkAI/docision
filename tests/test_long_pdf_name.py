"""Regression: file labels must not change PDF dispatch or exported filenames."""
import base64
import io
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from docision.local_app.runtime import score_file

class LongPDFNameTests(unittest.TestCase):
    def test_long_pdf_name_preserves_pages_and_full_filename(self):
        raw=io.BytesIO()
        first=Image.new('RGB',(50,50),'white');second=Image.new('RGB',(50,50),'gray')
        first.save(raw,format='PDF',save_all=True,append_images=[second])
        class Engine:
            def score_document(self,document):
                with Image.open(document['image']) as image: image.verify()
                return [{'decision':{'text':'Yes'}}]
        with tempfile.TemporaryDirectory() as directory:
            for name in ['short.pdf','a'*241+'.pdf','b'*241+'.PDF']:
                with self.subTest(name_length=len(name),suffix=name[-4:]):
                    output=score_file({'name':name,'data':base64.b64encode(raw.getvalue()).decode(),
                        'questions':[{'question':'Table?','candidates':['Yes','No']}]},Engine(),Path(directory))
                    self.assertEqual([r['page'] for r in output['rows']],[1,2])
                    self.assertTrue(all('result' in r and 'error' not in r for r in output['rows']))
                    self.assertTrue(all(r['file']==name for r in output['rows']))

if __name__=='__main__':unittest.main()
