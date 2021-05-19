import pandas as pd
import numpy as np
import requests
import re
import os
import zipfile
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

masterfile = pd.read_csv('master.csv')

def fraud_detection(url):
    r = requests.get(url)
    file_name = re.split(".zip", re.split("/", url)[-1])[0]
    known_images = []
    for i in os.listdir('identityPics-custID_PicID'):
        known_images.append(re.split("_", i))
    KEY = "6eaf5990fdca48cf84b2988af5f39fec"
    ENDPOINT = "https://whyiseverynametakenalready.cognitiveservices.azure.com/"
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
    acct_id = []
    bank_acct_id = []
    pictures = zipfile.ZipFile(BytesIO(r.content))
    verify = []
    for picture in pictures.namelist():
        pic_id = re.split("_", picture)[0]
        pic_id2 = re.split("[.]", re.split("_", picture)[1])[0]
        acct_id.append(pic_id)
        bank_acct_id.append(pic_id2)
        image = face_client.face.detect_with_stream(pictures.open(picture, "r"), detection_model='detection_03')
        image_id = image[0].face_id
        comparables = []
        for h in range(0, len(known_images)):
            if known_images[h][0] == re.split('_', picture)[0]:
                comparables.append(known_images[h][1])
        if comparables:
            compare = []
            for j in range(0, len(comparables)):
                compare.append(face_client.face.detect_with_stream(
                    open("identityPics-custID_PicID/" + pic_id + "_" + comparables[j], "rb"),
                    detection_model='detection_03')[0])

            compare_id = list(map(lambda x: x.face_id, compare))
            similar_faces = face_client.face.find_similar(face_id=image_id, face_ids=compare_id)
            if similar_faces:
                verify.append(1)
            else:
                verify.append(0)
    acct_id = pd.DataFrame(data={'acctID': acct_id, 'bankAcctID': bank_acct_id, 'verified': verify})
    acct_id['acctID'] = pd.to_numeric(acct_id['acctID'])
    acct_id['bankAcctID'] = pd.to_numeric(acct_id['bankAcctID'])
    acct_id2 = pd.merge(acct_id, masterfile, left_on=acct_id['acctID'], right_on=masterfile['custID'], how='left')
    acct_id2['falseAcct'] = np.where(acct_id2.bankAcctID == acct_id2.bankAcctID2, 0, 1)
    acct_id2['date'] = np.where((acct_id2.fraud == 1) | (acct_id2.falseAcct == 1) | (acct_id2.verified == 0), pd.NaT, acct_id2.date)
    acct_id2['date'] = acct_id2['date'].apply(pd.to_datetime)
    final = acct_id2[['acctID', 'date']]
    final['date'] = final['date'].astype(str)
    final['date'] = np.where(final.date == 'NaT', 'NA', final.date)
    return final, file_name


app = dash.Dash(prevent_initial_callbacks=True)
colors = {
    'background': 'white',
    'text': 'black'



}
app.layout = html.Div([
        html.H6("Please enter your URL below!"),
        html.Div(["URL: ",
                  dcc.Input(id='my-input', placeholder='URL', type='text')]),
        html.Button(id='submit-button-state', children='Submit', disabled=False),
        html.Br(),
        html.Div(id='my-output'),
        html.Button("Download CSV", id="btn_csv",disabled=True),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Store(id='dataframe'),
        dcc.Store(id='filename')
        ])

@app.callback(
            Output('my-output', 'children'),
            Input('submit-button-state', 'n_clicks')
            )

def note(n_clicks):
        if n_clicks > 0:
            return 'Please wait for the file to be generated. (This may take up to one minute)'
        else:
            return 'Please enter a URL.'

@app.callback(
    Output('dataframe', 'data'),
    Output('filename', 'data'),
    Output('btn_csv', 'disabled'),
    Input('submit-button-state', 'n_clicks'),
    State('my-input', 'value')
)

def generate_url(n_clicks, input_value):
    dataframe, filename = fraud_detection(input_value)
    return dataframe.to_dict('records'), filename, False


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    Input("dataframe", 'data'),
    Input("filename", 'data')
)

def download(n_clicks, final, file_name):
    if n_clicks:
        return dcc.send_data_frame(pd.DataFrame(final).to_csv,'{}.csv'.format(file_name),index=False)

if __name__ == '__main__':
    app.run_server(debug=True)



# fraud_detection('https://www.dropbox.com/s/30w0gzywa05uidz/5189798.zip?dl=1')