import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree

import matplotlib.pyplot as plt
import seaborn as sns

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Set pandas options so will print whole df

# load data from cvs file into pandas dataframe
dataset_1 = pd.read_csv(r"C:\Users\karan\PycharmProjects\RandomForestClassification\churn-bigml-80_graphs_Extended_2.csv")

# State Counts
state_data = pd.DataFrame(dataset_1.State.value_counts())
state_data_count = pd.DataFrame(state_data)
state_data_count_df = state_data_count.reset_index()
state_data_count_df.columns = ['state', 'counts'] # change column names

# International Plan Count
inter_plan_data = dataset_1['International plan'].value_counts()
inter_plan_data_count = pd.DataFrame(inter_plan_data)
inter_plan_data_df = inter_plan_data_count.reset_index()
inter_plan_data_df.columns = ['International Plan', 'Count']



state_fig = px.bar(state_data_count_df, x="state", y="counts", color='state')
iner_plan_fig = px.pie(inter_plan_data_df, values='Count', names='International Plan')



dcc.Graph(figure=state_fig)
dcc.Graph(figure=iner_plan_fig)

app = dash.Dash()
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H3('Column 1'),
            dcc.Graph(figure=state_fig)
        ], className="six columns"),

        html.Div([
            html.H3('Column 2'),
            dcc.Graph(figure=iner_plan_fig)
        ], className="six columns"),
    ], className="row")
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)