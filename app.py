from pathlib import Path
from io import StringIO
import hashlib
import json
import pandas as pd
import streamlit as st
from bank_ml import FEATURES, CATEGORICAL, NUMERIC, load_bundle, predict
from reports import report_context, html_report, pdf_report

st.set_page_config(page_title='Bank Data | AfriAI Solutions',page_icon='🏦',layout='wide')
st.title('Prédiction de souscription bancaire')
st.caption('AfriAI Solutions | Karaboué VAKABOU | Projet pédagogique')
st.write('Estimer la souscription à partir du profil client et des caractéristiques du contact.')
ROOT=Path(__file__).resolve().parent

@st.cache_resource
def load_model(signature):
    return load_bundle(ROOT)

try:
    # La signature déclenche un nouveau chargement après une nouvelle exécution du notebook.
    signature=hashlib.sha256((ROOT/'artifacts/metadata.json').read_bytes()).hexdigest()
    pipeline,meta=load_model(signature)
except Exception as e:
    st.error(f'Modèle indisponible : {e}')
    st.info('Exécuter entièrement le notebook dans ce même environnement, puis relancer Streamlit.')
    st.stop()

st.info('La durée du contact doit déjà être connue. Ce modèle ne permet pas un ciblage entièrement avant appel.')
with st.expander('Modèle, performances et limites'):
    st.write('Modèle retenu :',meta['selected_model'])
    st.write('Entraînement UTC :',meta['created_utc'])
    st.dataframe(pd.DataFrame([meta['test_metrics']],index=['Test du notebook']))
    for limitation in meta['limitations']:st.write('- '+limitation)
    st.caption('Performances issues des métadonnées du notebook ; probabilités non calibrées.')

LABELS={'age':'Âge','job':'Profession','marital':'Situation matrimoniale','education':'Niveau d’études',
        'housing':'Crédit immobilier','loan':'Crédit personnel','contact':'Type de contact','month':'Mois du contact',
        'day_of_week':'Jour du contact','duration':'Durée du contact (secondes)','campaign':'Nombre de contacts de la campagne',
        'pdays':'Jours depuis le contact précédent (999 = jamais contacté)','previous':'Contacts avant cette campagne',
        'poutcome':'Résultat de la campagne précédente'}
DEFAULTS={'age':35,'duration':180,'campaign':1,'pdays':999,'previous':0}
MODALITIES=dict(zip(CATEGORICAL,pipeline.named_steps['prepare'].uniques_))
example=pd.DataFrame([{'age':35,'job':'admin.','marital':'married','education':'university.degree',
'housing':'yes','loan':'no','contact':'cellular','month':'may','day_of_week':'mon','duration':180,
'campaign':1,'pdays':999,'previous':0,'poutcome':'nonexistent'}])[FEATURES]


def store_result(key,data):
    result=predict(data,pipeline)
    ctx=report_context(meta)
    st.session_state[key]={'result':result,'context':ctx,'html':html_report(result,ctx),'pdf':pdf_report(result,ctx)}


def show_result(key):
    if key not in st.session_state:return
    item=st.session_state[key];r=item['result']
    st.success(f"{len(r)} client(s) | Modèle : {meta['selected_model']}")
    if len(r)==1:
        a,b=st.columns(2)
        a.metric('Souscription prédite',r.iloc[0]['prediction_souscription'])
        b.metric('Probabilité estimée de souscription',f"{r.iloc[0]['probabilite_souscription_pct']:.2f} %")
    st.dataframe(r,hide_index=True)
    a,b,c=st.columns(3)
    a.download_button('Résultats CSV',r.to_csv(index=False).encode('utf-8-sig'),f'{key}.csv','text/csv',key=key+'csv')
    b.download_button('Rapport HTML',item['html'],f'{key}.html','text/html',key=key+'html')
    c.download_button('Rapport PDF',item['pdf'],f'{key}.pdf','application/pdf',key=key+'pdf')

# Une nouvelle version du modèle invalide les résultats conservés en session.
if st.session_state.get('model_signature')!=signature:
    for k in ['individual_result','batch_result','batch_hash']:st.session_state.pop(k,None)
    st.session_state['model_signature']=signature

simple,batch=st.tabs(['Prédiction individuelle','Prédictions par CSV'])
with simple:
    with st.form('client',clear_on_submit=False):
        cols=st.columns(2);values={}
        for i,c in enumerate(FEATURES):
            with cols[i%2]:
                if c in CATEGORICAL:values[c]=st.selectbox(LABELS[c],list(MODALITIES[c]))
                else:values[c]=st.number_input(LABELS[c],min_value=17 if c=='age' else (1 if c=='campaign' else 0),
                    max_value=120 if c=='age' else None,value=DEFAULTS[c],step=1)
        submit=st.form_submit_button('Prédire',type='primary')
    if submit:
        st.session_state.pop('individual_result',None)
        try:store_result('individual_result',pd.DataFrame([values])[FEATURES])
        except Exception as e:st.error(str(e))
    show_result('individual_result')
    if st.button('Nouvelle prédiction'):
        st.session_state.pop('individual_result',None);st.rerun()

with batch:
    st.write('Colonnes requises, ordre libre :');st.code(','.join(FEATURES))
    st.caption('Valeurs manquantes rejetées ; catégories inédites signalées. Limite de cette interface : 1 000 clients par lot.')
    st.download_button('Exemple CSV fictif',example.to_csv(index=False).encode('utf-8-sig'),'exemple_clients.csv','text/csv')
    epoch=st.session_state.get('upload_epoch',0)
    uploaded=st.file_uploader('Choisir le CSV',type=['csv'],key=f'upload_{epoch}')
    current_hash=hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded else None
    if current_hash!=st.session_state.get('batch_hash'):
        st.session_state.pop('batch_result',None);st.session_state['batch_hash']=current_hash
    if uploaded:
        try:
            text=uploaded.getvalue().decode('utf-8-sig')
            from csv import Sniffer
            dialect=Sniffer().sniff(text[:8192],delimiters=',;\t')
            from csv import reader
            header=next(reader(StringIO(text),dialect))
            if len(set(c.strip() for c in header))!=len(header):raise ValueError('Colonnes dupliquées dans le CSV.')
            data=pd.read_csv(StringIO(text),sep=dialect.delimiter)
            data.columns=data.columns.str.strip()
            if len(data)>1000:raise ValueError('Maximum 1 000 clients par fichier.')
            st.dataframe(data.head(10),hide_index=True)
            if st.button('Prédire pour tous les clients',type='primary'):
                st.session_state.pop('batch_result',None)
                with st.spinner('Calcul et préparation des rapports…'):store_result('batch_result',data)
        except Exception as e:st.error(str(e))
    show_result('batch_result')
    if st.button('Recommencer le lot'):
        st.session_state.pop('batch_result',None)
        st.session_state['upload_epoch']=epoch+1;st.rerun()
