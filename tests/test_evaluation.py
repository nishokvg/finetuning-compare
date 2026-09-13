import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate import LABELS,normalize,summarize

class EvaluationTests(unittest.TestCase):
    def test_class_names_do_not_follow_sklearn_sort_order(self):
        rows=[{'label':label,'prediction':label,'seconds':1} for label in reversed(LABELS)]
        # Deliberately confuse Computer-Services with O365; verify the exact named class.
        rows[0]['prediction']='O365'
        report=summarize(rows)
        self.assertEqual(report['classification_report']['Computer-Services']['recall'],0)
        self.assertEqual(report['classification_report']['Active Directory']['recall'],1)
        self.assertEqual(report['confusion_matrix'][6][2],1)

    def test_invalid_prediction_is_not_a_general_ticket(self):
        result=summarize([{'label':'Support general','prediction':normalize('AD'),'seconds':1}])
        self.assertEqual(result['accuracy'],0)
        self.assertEqual(result['invalid_outputs'],1)
        self.assertEqual(result['confusion_matrix'][0][7],1)

    def test_formatting_is_tolerated_but_explanations_are_not(self):
        self.assertEqual(normalize('  active DIRECTORY\n'),'Active Directory')
        self.assertEqual(normalize('O365 because it concerns email'),'INVALID')

if __name__=='__main__': unittest.main()
