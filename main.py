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
from dash.dependencies import Input, Output

# Set pandas options so will print whole df
pd.set_option("display.max_rows", None, "display.max_columns", None)

# dataset = np.loadtxt(r"C:\Users\karan\PycharmProjects\RandomForestClassification\churn-bigml-80_cleaned.csv", delimiter=",")
# print(dataset.shape)

# load data from cvs file into pandas dataframe
data = pd.read_csv(r"C:\Users\karan\PycharmProjects\RandomForestClassification\churn-bigml-80_cleaned.csv")

# one hot encode satates
values = pd.get_dummies(data.state_numeric)
print(values)


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
print(df)

### To slice df so not include states
##df = df.iloc[:21]

# Show column names
for x in index_list:
    print(f"column names: {x}")

# Plotly set up w/ data
ALLOWED_TYPES = (
    "text", "number", "password", "email", "search",
    "tel", "url", "range", "hidden",
)

fig = px.bar(df, x='feature', y='score')
app = dash.Dash()

app.layout = html.Div([

    html.Div([

        html.Div([
            html.H5("Area Code: "),
            dcc.Input(id='area_code_input', value='10', type="number")],
            ),

        html.Div([
            html.H5("Account Length: "),
            dcc.Input(id='acc_len_input', value='10', type="number")],
            ),

        html.Div([
            html.H5("International Plan: "),
            dcc.Input(id='int_plan_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Voicemail Plan: "),
            dcc.Input(id='vmail_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Number of Voicemails: "),
            dcc.Input(id='vmail_num_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Day Mins: "),
            dcc.Input(id='day_mins_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Day Calls: "),
            dcc.Input(id='day_calls_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Day Charges: "),
            dcc.Input(id='day_charges_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Evening Mins: "),
            dcc.Input(id='eve_mins_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Evening Calls: "),
            dcc.Input(id='eve_calls_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Evening Charges: "),
            dcc.Input(id='eve_charges_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Night Mins: "),
            dcc.Input(id='night_mins_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Night Calls: "),
            dcc.Input(id='night_calls_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Night Charges: "),
            dcc.Input(id='night_charges_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Inter Mins: "),
            dcc.Input(id='inter_mins_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Inter Calls: "),
            dcc.Input(id='inter_calls_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Total Inter Charges: "),
            dcc.Input(id='inter_charges_input', value='1', type="number")],
            ),

        html.Div([
            html.H5("Customer service calls: "),
            dcc.Input(id='cs_calls_input', value='1', type="number")],
            )


        ], style={'display': 'grid',
                  'grid-template-columns': 'auto auto auto auto',
                  'grid-row-gap': '5px',
                  'background-color': '#2196F3',
                  'padding': '5px'}),

    html.Br(),

    html.Div(id='my_output', style={'width': '300px',
                                    'border': '15px solid green ',
                                    'padding': '50px',
                                    'margin': '20px'}),

    html.Div([dcc.Graph(figure=fig)]), ],
    id="styled-numeric-input"
)


@app.callback(
    Output(component_id='my_output', component_property='children'),

    Input(component_id='area_code_input', component_property='value'),
    Input(component_id='acc_len_input', component_property='value'),

    Input(component_id='int_plan_input', component_property='value'),

    Input(component_id='vmail_input', component_property='value'),
    Input(component_id='vmail_num_input', component_property='value'),

    Input(component_id='day_mins_input', component_property='value'),
    Input(component_id='day_calls_input', component_property='value'),
    Input(component_id='day_charges_input', component_property='value'),

    Input(component_id='eve_mins_input', component_property='value'),
    Input(component_id='eve_calls_input', component_property='value'),
    Input(component_id='eve_charges_input', component_property='value'),

    Input(component_id='night_mins_input', component_property='value'),
    Input(component_id='night_calls_input', component_property='value'),
    Input(component_id='night_charges_input', component_property='value'),

    Input(component_id='inter_mins_input', component_property='value'),
    Input(component_id='inter_calls_input', component_property='value'),
    Input(component_id='inter_charges_input', component_property='value'),
    Input(component_id='cs_calls_input', component_property='value')
)





def update_output_div(area_code, acc_len, int_plan, vmail, vmail_num,
                      day_mins, day_calls, day_charge,
                      eve_mins, eve_calls, eve_charge,
                      night_mins, night_calls, night_change,
                      inter_mins, inter_calls, inter_change,cs_calls):

    all_mins = int(day_mins) + int(eve_mins) + int(night_mins) + int(inter_mins)
    all_calls = int(day_calls) + int(eve_calls) + int(night_calls) + int(inter_calls)
    all_charges =  int(day_charge) + int(eve_charge) + int(night_change) + int(inter_change)


    test_pred = clf.predict([[area_code, acc_len, int_plan, vmail, vmail_num,
                              day_mins, day_calls, day_charge,
                              eve_mins, eve_calls, eve_charge,
                              night_mins, night_calls, night_change,
                              inter_mins, inter_calls, inter_change,
                              all_mins, all_calls, all_charges, cs_calls,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print(f"Logged output: {test_pred}")
    return 'Output: {}'.format(test_pred)


# To run return
app.run_server(debug=True, use_reloader=False)
