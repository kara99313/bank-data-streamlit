"""Contrôles reproductibles : python -m unittest -v test_bank.py (après notebook)."""
import unittest
from pathlib import Path
from io import BytesIO
import json
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from bank_ml import FEATURES, BankPreprocessor, validate_input, predict, load_bundle
from reports import report_context, html_report, pdf_report

ROOT=Path(__file__).resolve().parent

class BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe,cls.meta=load_bundle(ROOT)
        cls.x=pd.read_csv(ROOT/'data/Bank_Data.csv')[FEATURES].head(50)

    def test_batch_individual_equivalence(self):
        batch=predict(self.x,self.pipe)
        for i in [0,15,49]:
            single=predict(self.x.iloc[[i]],self.pipe)
            self.assertEqual(single.prediction_souscription.iloc[0],batch.prediction_souscription.iloc[i])
            self.assertEqual(single.probabilite_souscription_pct.iloc[0],batch.probabilite_souscription_pct.iloc[i])

    def test_column_order(self):
        pd.testing.assert_frame_equal(predict(self.x,self.pipe)[['prediction_souscription','probabilite_souscription_pct']],
            predict(self.x[FEATURES[::-1]],self.pipe)[['prediction_souscription','probabilite_souscription_pct']])

    def test_missing_column(self):
        with self.assertRaises(ValueError):predict(self.x.drop(columns='age'),self.pipe)

    def test_missing_value(self):
        x=self.x.copy();x.loc[0,'job']=None
        with self.assertRaises(ValueError):predict(x,self.pipe)

    def test_nonfinite(self):
        x=self.x.astype({'duration':float});x.loc[0,'duration']=np.inf
        with self.assertRaises(ValueError):predict(x,self.pipe)

    def test_negative(self):
        x=self.x.copy();x.loc[0,'duration']=-1
        with self.assertRaises(ValueError):predict(x,self.pipe)

    def test_empty(self):
        with self.assertRaises(ValueError):predict(self.x.iloc[:0],self.pipe)

    def test_duplicate_column(self):
        x=pd.concat([self.x,self.x[['age']]],axis=1)
        with self.assertRaises(ValueError):predict(x,self.pipe)

    def test_unknown_category_signaled(self):
        x=self.x.head(1).copy();x.loc[0,'job']='INCONNU_TEST'
        self.assertIn('catégorie inconnue',predict(x,self.pipe).traitements_signales.iloc[0])

    def test_input_unchanged(self):
        x=self.x.copy(deep=True);predict(x,self.pipe);pd.testing.assert_frame_equal(x,self.x)

    def test_no_refit_on_prediction(self):
        before=joblib.hash(self.pipe);predict(self.x,self.pipe)
        self.assertEqual(before,joblib.hash(self.pipe))

    def test_preparation_independent_of_target(self):
        # Ajuster sur un petit train ; la cible fournie ne doit rien changer.
        p1=BankPreprocessor().fit(self.x,np.zeros(len(self.x)))
        p2=BankPreprocessor().fit(self.x,np.ones(len(self.x)))
        self.assertEqual(joblib.hash(p1),joblib.hash(p2))

    def test_holdout_not_fitted(self):
        prep=BankPreprocessor().fit(self.x)
        before=joblib.hash(prep);x=self.x.head(1).copy();x.loc[0,'duration']=99999
        prep.transform(x);self.assertEqual(before,joblib.hash(prep))
        self.assertIn('duration',prep.flags(x)[0])

    def test_special_pdays_preserved(self):
        x=self.x.head(1).copy();x.loc[0,'pdays']=999
        transformed=self.pipe.named_steps['prepare'].transform(x)
        self.assertEqual(transformed[0,FEATURES.index('pdays')],999)

    def test_separate_objects(self):
        _,_,s,m=joblib.load(ROOT/'artifacts/list_items.joblib')
        p=joblib.load(ROOT/'artifacts/preprocessor.joblib')
        np.testing.assert_array_equal(m.predict(s.transform(p.transform(self.x))),self.pipe.predict(self.x))
        np.testing.assert_allclose(m.predict_proba(s.transform(p.transform(self.x))),self.pipe.predict_proba(self.x),atol=0,rtol=0)

    def test_reports(self):
        result=predict(self.x,self.pipe);ctx=report_context(self.meta)
        html=html_report(result,ctx).decode()
        self.assertIn(self.meta['selected_model'],html);self.assertIn(ctx['reference'],html)
        pdf=pdf_report(result,ctx)
        text='\n'.join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
        self.assertIn(self.meta['selected_model'],text);self.assertIn('50',text)
        self.assertIn('Ligne',text);self.assertIn('50',text)

    def test_html_escape(self):
        result=predict(self.x.head(1),self.pipe);result.loc[0,'job']='<script>alert(1)</script>'
        html=html_report(result,report_context(self.meta)).decode()
        self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)

if __name__=='__main__': unittest.main()
