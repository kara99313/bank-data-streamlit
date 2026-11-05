"""Tests de l'interface Streamlit avec AppTest, sans navigateur public."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class InterfaceTests(unittest.TestCase):
    def test_individual_and_download_persistence(self):
        app=AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=90).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        app.button(key='FormSubmitter:client-Prédire').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertIn('individual_result',app.session_state)
        self.assertEqual(len(app.session_state['individual_result']['result']),1)
        app.run()
        self.assertIn('individual_result',app.session_state)
        next(b for b in app.button if b.label=='Nouvelle prédiction').click().run()
        self.assertNotIn('individual_result',app.session_state)
        self.assertEqual(len(app.exception),0)

if __name__=='__main__':unittest.main()
