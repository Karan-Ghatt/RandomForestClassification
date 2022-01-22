import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash import dash_table
import plotly.express as px
from dash.dependencies import Input, Output

#### RANDOMFOREST

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Set pandas options so will print whole df
pd.set_option("display.max_rows", None, "display.max_columns", None)

# dataset = np.loadtxt(r"C:\Users\karan\PycharmProjects\RandomForestClassification\churn-bigml-80_cleaned.csv", delimiter=",")
# print(dataset.shape)

# load data from cvs file into pandas dataframe
data = pd.read_csv(r"C:\Users\karan\PycharmProjects\RandomForestClassification\churn-bigml-80_cleaned.csv")

# one hot encode satates
values = pd.get_dummies(data.state_numeric)



# append to original dataset - new df created
new_df = pd.concat([data, values], axis=1)
# columns of new datafram
new_df_columns = [*new_df]
# copy of columns of new datafram to remove churn_numeric and state_numeric from train data
new_df_columns_2 = [*new_df]
new_df_columns_2.remove('churn_numeric')
new_df_columns_2.remove('state_numeric')


# list of column headers, minius churn and state numeric - used to pass to feature importance
index_list = new_df_columns_2

# spliting data into features and labeles (values and outcome)
x_data = new_df[new_df_columns_2]
print([*x_data])

y_data = new_df[['churn_numeric']]  # lables

# splitting dataset into tarining and testing set - 70% training, 30% testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# create goussian classifier
clf = RandomForestClassifier(n_estimators=100)

# train modle using training datasets
clf.fit(x_train, y_train)

# prediction on test dataset
y_pred = clf.predict(x_test)

# Modle accuracy; how offten is the classifier correct
print(f"Accuracy: {(metrics.accuracy_score(y_test, y_pred)) * 100}%")
###NEED TO MAKE INTO FUNCTION SO CAN USE IN DASH - I THINK??
##testing vales for modle predicition
# test_pred = clf.predict([[17, 408, 77, 0, 0, 0,
#                   62.4, 89, 10.61, 169.9,
#                   121, 14.44, 209.6, 64,
#                   9.43, 5.7, 6, 1.54, 447.6,
#                   280, 36.02, 5]])


# def testings(listyboi):
#     return clf.predict([listyboi])
#     pass


# feature importance weighting
feature_imp = pd.Series(clf.feature_importances_, index=index_list).sort_values(ascending=False)
##print(feature_imp)


# #vizualisation of feature importance
# sns.barplot(x=feature_imp, y = feature_imp.index)
# plt.xticks(rotation=45)
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Feature')
# plt.legend()
# plt.show()


# vizualisation of tree
# estimator = clf.estimators_[2]
# plt.show()



# TO USE DASH
# Have generated feature importance output and saved as cvs
# Read cvs file as dataframe and add column headers
df = pd.read_csv(r"C:\Users\karan\PycharmProjects\RandomForestClassification\testing_file_one.csv",
                 names=['feature', 'score'])

# Show datafram


# To slice df so not include states
df = df.iloc[:21]

# Show column names
#for x in index_list:
#    print(f"column names: {x}")




# Data Graphs
# Dataset for analysis
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

# Vmail Plan Count
vmail_plan_data = dataset_1['Voice mail plan'].value_counts()
vmail_plan_data_count = pd.DataFrame(vmail_plan_data)
vmail_plan_data_df = vmail_plan_data_count.reset_index()
vmail_plan_data_df.columns = ['Voice Mail Plan', 'Count']



# Churn Count
churn_data = dataset_1['Churn'].value_counts()
churn_data_count = pd.DataFrame(churn_data)
churn_data_df = churn_data_count.reset_index()
churn_data_df.columns = ['Churn', 'Count']


# Output figure
fig = px.bar(df, x='feature', y='score',
             color='feature',
             title='Feature Importance Score by Feature')
fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
fig.update_layout(font_color="white")


state_fig = px.bar(state_data_count_df, x="state", y="counts",
                   color='state',
                   title="Count by State")
state_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
state_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
state_fig.update_layout(font_color="white")



iner_plan_fig = px.pie(inter_plan_data_df, values='Count', names='International Plan',
                       title='International Plan')
iner_plan_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
iner_plan_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
iner_plan_fig.update_layout(font_color="white")



vmail_plan_fig = px.pie(vmail_plan_data_df, values='Count', names='Voice Mail Plan',
                        title='Voicemail Plan')
vmail_plan_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
vmail_plan_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
vmail_plan_fig.update_layout(font_color="white")


churn_fig = px.pie(churn_data_df, values='Count', names='Churn', title='Churn')
churn_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
churn_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
churn_fig.update_layout(font_color="white")


all_min_fig = px.histogram(dataset_1, x='Tota all miniutes')
all_min_fig.update_layout(bargap=0.1)
all_min_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
all_min_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
all_min_fig.update_layout(font_color="white")


all_call_fig = px.histogram(dataset_1, x='Total all calls')
all_call_fig.update_layout(bargap=0.1)
all_call_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
all_call_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
all_call_fig.update_layout(font_color="white")


all_charge_fig = px.histogram(dataset_1, x='Total all charges')
all_charge_fig.update_layout(bargap=0.1)
all_charge_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
all_charge_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
all_charge_fig.update_layout(font_color="white")


acc_len_fig = px.histogram(dataset_1, x='Account length')
acc_len_fig.update_layout(bargap=0.1)
acc_len_fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
acc_len_fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
acc_len_fig.update_layout(font_color="white")



# Getting mean data
all_avg_data_frame = dataset_1.groupby('Churn').mean().round(2)
print(all_avg_data_frame)
avg_data_frame = all_avg_data_frame[['Account length', 'Number vmail messages',
                                     'Tota all miniutes', 'Total all calls',
                                     'Total all charges', 'Customer service calls']]




### CREATING DASH APP

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

navbar = dbc.Navbar(
    dbc.Container([
            html.A(
                html.Header([
                    html.H4("Customer Churn Dashboard")]
                )
            ),
        ]),
    color="dark",
    dark=True,
)


card_one = dbc.Card([
    dbc.CardHeader("Area Code"),
    dbc.CardBody([
        dcc.Dropdown(id='area_code_input',
                     style={'color': 'black'},
                     options=[{'label': '415', 'value': '415'},
                              {'label': '408', 'value': '408'},
                              {'label': '510', 'value':'510'},
                              ],
                     value='')
    ])
])

card_two = dbc.Card([
    dbc.CardHeader("Account Length"),
    dbc.CardBody([
        dcc.Input(id='account_len_input', value='10', type="number")
    ])
])

card_three = dbc.Card([
    dbc.CardHeader("International Plan"),
    dbc.CardBody([
        dcc.Dropdown(id='international_plan_input',
                     style={'color': 'black'},
                     options=[{'label': 'Yes', 'value': '1'},
                              {'label': 'No', 'value': '0'}
                              ],
                     value='')
    ])
])

card_four = dbc.Card([
    dbc.CardHeader("VoiceMail Plan"),
    dbc.CardBody([
        dcc.Dropdown(id='voicemail_plan_input',
                     style={'color': 'black'},
                     options=[{'label': 'Yes', 'value': '1'},
                              {'label': 'No', 'value': '0'}
                              ],
                     value='')
    ])
])

card_five = dbc.Card([
    dbc.CardHeader("Customer Service Calls"),
    dbc.CardBody([
        dcc.Input(id='customer_service_calls_input', value='10', type="number")
    ])
])

card_six = dbc.Card([
    dbc.CardHeader("State"),
    dbc.CardBody([
        dcc.Dropdown(id='state_input',
                     style={'color': 'black'},
                     options=[{'label': 'KS', 'value': '1'},
                                {'label': 'OH', 'value': '2'},
                                {'label': 'NJ', 'value': '3'},
                                {'label': 'OK', 'value': '4'},
                                {'label': 'AL', 'value': '5'},
                                {'label': 'MA', 'value': '6'},
                                {'label': 'MO', 'value': '7'},
                                {'label': 'WV', 'value': '8'},
                                {'label': 'RI', 'value': '9'},
                                {'label': 'IA', 'value': '10'},
                                {'label': 'MT', 'value': '11'},
                                {'label': 'ID', 'value': '12'},
                                {'label': 'VT', 'value': '13'},
                                {'label': 'VA', 'value': '14'},
                                {'label': 'TX', 'value': '15'},
                                {'label': 'FL', 'value': '16'},
                                {'label': 'CO', 'value': '17'},
                                {'label': 'AZ', 'value': '18'},
                                {'label': 'NE', 'value': '19'},
                                {'label': 'WY', 'value': '20'},
                                {'label': 'IL', 'value': '21'},
                                {'label': 'NH', 'value': '22'},
                                {'label': 'LA', 'value': '23'},
                                {'label': 'GA', 'value': '24'},
                                {'label': 'AK', 'value': '25'},
                                {'label': 'MD', 'value': '26'},
                                {'label': 'AR', 'value': '27'},
                                {'label': 'WI', 'value': '28'},
                                {'label': 'OR', 'value': '29'},
                                {'label': 'DE', 'value': '30'},
                                {'label': 'IN', 'value': '31'},
                                {'label': 'UT', 'value': '32'},
                                {'label': 'CA', 'value': '33'},
                                {'label': 'SD', 'value': '34'},
                                {'label': 'NC', 'value': '35'},
                                {'label': 'WA', 'value': '36'},
                                {'label': 'MN', 'value': '37'},
                                {'label': 'NM', 'value': '38'},
                                {'label': 'NV', 'value': '39'},
                                {'label': 'DC', 'value': '40'},
                                {'label': 'NY', 'value': '41'},
                                {'label': 'KY', 'value': '42'},
                                {'label': 'ME', 'value': '43'},
                                {'label': 'MS', 'value': '44'},
                                {'label': 'MI', 'value': '45'},
                                {'label': 'SC', 'value': '46'},
                                {'label': 'TN', 'value': '47'},
                                {'label': 'PA', 'value': '48'},
                                {'label': 'HI', 'value': '49'},
                                {'label': 'ND', 'value': '50'},
                                {'label': 'CT', 'value': '51'}
                                ],
                     value='')
    ])
])

card_seven = dbc.Card([
    dbc.CardHeader("Total Day Minutes"),
    dbc.CardBody([
        dcc.Input(id='total_day_min_input', value='10', type="number")
    ])
])

card_eight = dbc.Card([
    dbc.CardHeader("Total Day Calls"),
    dbc.CardBody([
        dcc.Input(id='total_day_calls_input', value='10', type="number")
    ])
])

card_nine = dbc.Card([
    dbc.CardHeader("Total Day Charges"),
    dbc.CardBody([
        dcc.Input(id='total_day_charges_input', value='10', type="number")
    ])
])

card_ten = dbc.Card([
    dbc.CardHeader("Total Evening Minutes"),
    dbc.CardBody([
        dcc.Input(id='total_eve_min_input', value='10', type="number")
    ])
])

card_eleven = dbc.Card([
    dbc.CardHeader("Total Evening Calls"),
    dbc.CardBody([
        dcc.Input(id='total_eve_calls_input', value='10', type="number")
    ])
])

card_twelve = dbc.Card([
    dbc.CardHeader("Total Evening Charges"),
    dbc.CardBody([
        dcc.Input(id='total_eve_charges_input', value='10', type="number")
    ])
])

card_thirteen = dbc.Card([
    dbc.CardHeader("Total Night Minutes"),
    dbc.CardBody([
        dcc.Input(id='total_night_min_input', value='10', type="number")
    ])
])

card_fourteen = dbc.Card([
    dbc.CardHeader("Total Night Calls"),
    dbc.CardBody([
        dcc.Input(id='total_night_calls_input', value='10', type="number")
    ])
])

card_fifteen = dbc.Card([
    dbc.CardHeader("Total Night Charges"),
    dbc.CardBody([
        dcc.Input(id='total_night_charges_input', value='10', type="number")
    ])
])

card_sixteen = dbc.Card([
    dbc.CardHeader("Total International Minutes"),
    dbc.CardBody([
        dcc.Input(id='total_inter_min_input', value='10', type="number")
    ])
])

card_seventeen = dbc.Card([
    dbc.CardHeader("Total International Calls"),
    dbc.CardBody([
        dcc.Input(id='total_inter_calls_input', value='10', type="number")
    ])
])

card_eighteen = dbc.Card([
    dbc.CardHeader("Total International Charges"),
    dbc.CardBody([
        dcc.Input(id='total_inter_charges_input', value='10', type="number")
    ])
])

card_nineteen = dbc.Card([
    dbc.CardHeader("Number of Voicemails"),
    dbc.CardBody([
        dcc.Input(id='num_vMails', value='10', type="number")
    ])
])



output_card = dbc.Card([
    dbc.CardHeader("Churn Prediction"),
    dbc.CardBody([
        html.Div(id="my_output")]
    )],
    color="light",
    inverse=True,
    className="w-75 mb-3")




card_graph = dbc.Card(
    dcc.Graph(id='my_graph', figure=fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

state_graph_card = dbc.Card(
    dcc.Graph(id='state_graph', figure=state_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

inter_plan_graph_card = dbc.Card(
    dcc.Graph(id='inter_plan_graph', figure=iner_plan_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

vmail_plan_graph_card = dbc.Card(
    dcc.Graph(id='vmail_plan_graph', figure=vmail_plan_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

churn_plan_graph_card = dbc.Card(
    dcc.Graph(id='churn_data_graph', figure=churn_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

# All Mins
all_min_histogram = dbc.Card(
    dcc.Graph(id='all_min_histogram', figure=all_min_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

# All Calls
all_call_histogram = dbc.Card(
    dcc.Graph(id='all_call_histogram', figure=all_call_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

# All Charges
all_charge_histogram = dbc.Card(
    dcc.Graph(id='all_charge_histogram', figure=all_charge_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)

# Account Length
acc_len_histogram = dbc.Card(
    dcc.Graph(id='acc_len_histogram', figure=acc_len_fig),
    body=True ,
    color='secondary',
    className="w-85 mb-3"
)



app.layout = html.Div([

    html.Center([
        html.Div([navbar])
    ]),

    html.Br(),

    dbc.Tabs([
        dbc.Tab(
            dbc.Row([
                dbc.Col(card_graph)]),
            label='Feature Importance'
        ),
        dbc.Tab(
            dbc.Row([
                dbc.Col(inter_plan_graph_card),
                dbc.Col(vmail_plan_graph_card),
                dbc.Col(churn_plan_graph_card)]),
            label='Data Overview'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(state_graph_card)]),
            label= 'State Breakdown'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(all_min_histogram),
                dbc.Col(all_call_histogram),
                dbc.Col(all_charge_histogram),
                dbc.Col(acc_len_histogram)]),
            label= 'Histograms'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(dash_table.DataTable(
                    id = 'avg_table',
                    columns=[{"name": i, "id": i} for i in avg_data_frame.columns],
                    data=avg_data_frame.to_dict('records'),
                    style_data={'backgroundColor':'transparent'},
                    style_header={'backgroundColor':'transparent'}
                ))
            ]),
            label='Average Data Tables'
        )

    ]),

    html.Br(),

    dbc.Tabs([
        dbc.Tab(
            dbc.Row([
                dbc.Col(card_six, width=2),
                dbc.Col(card_one, width=2),
                dbc.Col(card_five, width=2),
                dbc.Col(card_two, width=2)]),
            label='General Data'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(card_four, width=2),
                dbc.Col(card_nineteen, width=2)]),
            label='Voicemail Data'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(card_seven, width=2),
                dbc.Col(card_eight, width=2),
                dbc.Col(card_nine, width=2)]),
            label='Day Data'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(card_ten, width=2),
                dbc.Col(card_eleven, width=2),
                dbc.Col(card_twelve, width=2)]),
            label='Evening Data'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(card_thirteen, width=2),
                dbc.Col(card_fourteen, width=2),
                dbc.Col(card_fifteen, width=2)]),
            label='Night Data'
        ),

        dbc.Tab(
            dbc.Row([
                dbc.Col(card_three, width=2),
                dbc.Col(card_sixteen, width=2),
                dbc.Col(card_seventeen, width=2),
                dbc.Col(card_eighteen, width=2)]),
            label='International Data'
        ),
    ]),

    html.Br(),

    html.Center([
        dbc.Row([
            dbc.Col(dbc.Button("Generate Prediction", id='button', n_clicks=None)),
            dbc.Col(dbc.Button('Reset Prediction', id='reset_button', n_clicks=0))
        ])
    ]),

    html.Br(),

    html.Center([
        dbc.Row([
            dbc.Col(output_card)])
    ]),

    html.Br()

])

@app.callback(
    Output(component_id='my_output', component_property='children'),

    Input(component_id='state_input', component_property='value'),
    Input(component_id='area_code_input', component_property='value'),
    Input(component_id='customer_service_calls_input', component_property='value'),
    Input(component_id='account_len_input', component_property='value'),


    Input(component_id='voicemail_plan_input', component_property='value'),
    Input(component_id='num_vMails', component_property='value'),

    Input(component_id='total_day_min_input', component_property='value'),
    Input(component_id='total_day_calls_input', component_property='value'),
    Input(component_id='total_day_charges_input', component_property='value'),

    Input(component_id='total_eve_min_input', component_property='value'),
    Input(component_id='total_eve_calls_input', component_property='value'),
    Input(component_id='total_eve_charges_input', component_property='value'),

    Input(component_id='total_night_min_input', component_property='value'),
    Input(component_id='total_night_calls_input', component_property='value'),
    Input(component_id='total_night_charges_input', component_property='value'),


    Input(component_id='international_plan_input', component_property='value'),
    Input(component_id='total_inter_min_input', component_property='value'),
    Input(component_id='total_inter_calls_input', component_property='value'),
    Input(component_id='total_inter_charges_input', component_property='value'),

    Input(component_id='button', component_property='n_clicks')
)

def update_output_div(state, area_code, cus_calls, account_len,
                      vmail_plan, num_vMails,
                      day_min, day_calls, day_charge,
                      eve_min, eve_calls, eve_change,
                      night_min, night_calls, night_charge,
                      int_plan, inter_min, inter_calls, inter_charge,
                      n):



    if n >0:

        state_list = [(x - x) for x in range(51)]
        state_list[int(state)-1] = 1

        print(state_list)

        all_min = int(day_min) + int(eve_min) + int(night_min) + int(inter_min)

        all_calls = int(day_calls) + int(eve_calls) + int(night_calls) + int(inter_calls)

        all_charge = int(day_charge) + int(eve_change) + int(night_charge) + int(inter_charge)




        var_arr = [int(area_code), int(account_len), int(int_plan),
                                  int(vmail_plan),
                                  int(num_vMails),
                                   int(day_min), int(day_calls), int(day_charge),
                                  int(eve_min), int(eve_calls), int(eve_change),
                                  int(night_min), int(night_calls), int(night_charge),
                                  int(inter_min), int(inter_calls), int(inter_charge),
                                  all_min, all_calls, all_charge, cus_calls]

        var_arr.extend(state_list)




        modle_pred = clf.predict([var_arr])

        # if = 1 then churn is true thus customer left

        if modle_pred == [0]:
            ret = ' not have'
        else:
            ret = ' have'


        return f'''The customer would {ret} left - Predicted  
        with an accuracy of {(metrics.accuracy_score(y_test, y_pred)) * 100}'''


@app.callback(Output('button','n_clicks'),
             [Input('reset_button','n_clicks')])
def update(reset):
    return 0


if __name__ == '__main__':
    app.run_server(debug=True)