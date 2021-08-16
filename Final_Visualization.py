import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly
import plotly.express as px
import dash_daq as daq
import math
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots


app = dash.Dash()

# Read the data from the Excel File.
dataFrame = pd.read_excel('CSE 332 - Database.xlsx')

# ---------------------------------------------CSS_Styles----------------------------------------------------

tabStyle = {
    'fontWeight': 'bold',
    'background': 'cornsilk',
    'border': 'white',
    'fontSize': 20
}

tabStyleWhenSelected = {
    'backgroundColor': 'grey',
    'color': 'white',
    'fontSize': 22,
    'borderTop': '2px solid white',
    'borderBottom': '2px solid white',
}

axisTitleStyle = {
    'text-align': 'center',
    'font-weight': 'bold',
    'color': 'white',
    'font-size': 22
}

# Legend style for histograms and scatter plots.
legendStyle = dict (
    title_font_family = 'Times New Roman',
    title_font_size = 18,
    title_font_color = 'black',
    font = dict (
        size = 14,
        color = 'black'
    ),
    bgcolor = 'LightSteelBlue',
    borderwidth = 2,
    bordercolor = 'black',
    yanchor = 'top',
    y = 1,
    xanchor = 'right',
    x = 1.3
)

# ---------------------------------------------Components and Variables----------------------------------------------------
labelDictionary = {
        'Population': 'Total Population',
        'Male_Pop': 'Male Population',
        'Female_Pop': 'Female Population',
        'Rent_Mean': 'Rent Mean',
        'Rent_Median': 'Rent Median',
        'Rent_Samples': 'Number of Rent Samples',
        'Rent_GT_25': '25% Greater Rent than Income',
        'Rent_GT_50': '50% Greater Rent than Income',
        'HI_Mean': 'Average Income',
        'HI_Median': 'Median Income',
        'HI_Samples': 'Number of Income Samples',
        'HC_Mortgage_Mean': 'Average Mortgage and Household Owner Costs',
        'HC_Mortgage_Median': 'Median Mortgage and Household Owner Costs',
        'HC_Mortgage_Samples': 'Number of Mortgage and Household Owner Costs Samples',
        'HC_Mean': 'Average Household Owner Costs',
        'HC_Median': 'Median Household Owner Costs',
        'HC_Samples': 'Number of Household Owner Costs Samples',
        'Home_Equity_Second_Mortgage': 'Second Mortgage and Home Equity Loans',
        'Second_Mortgage': 'Percent with Second Mortgage', 'value': 'Second_Mortgage',
        'Home_Equity': 'Percent with Home Equity Loans', 'value': 'Home_Equity',
        'Debt': 'Debt',
        'HS_Degree': 'Percent of Population who Passed High School',
        'HS_Degree_Male': 'Percent of Males who Passed High School',
        'HS_Degree_Female': 'Percent of Females who Passed High School',
        'Male_Age_Mean': 'Male Age Mean',
        'Male_Age_Median': 'Male Age Median',
        'Male_Age_Samples': 'Number of Male Age Samples',
        'Female_Age_Mean': 'Female Age Mean',
        'Female_Age_Median': 'Female Age Median',
        'Female_Age_Samples': 'Number of Female Age Samples',
        'Married': 'Percent Married',
        'Separated': 'Percent Separated',
        'Divorced': 'Percent Divorced',
        'State_Unemployment_Rate': 'State Unemployment Rate',
        'Area_Land': 'Area of Land',
        'Area_Water': 'Area of Water', 'value': 'Area_Water',
        'State_Mental_Health_Rank': 'State Mental Health Rank',
        'Histogram_ID': 'State + D.C. (1st Letter Ranges)',
        'Pie_Chart_ID': 'Ranges of Values',
        'UID': 'User ID',
        'State_ID': 'State ID'
}

# ----------------------------------------------------Layout---------------------------------------------

app.layout = html.Div(style = {'background-color': '#313131'}, children = [
    html.H1("USA Statistics", style = {'font-size': 35, 'margin': 'auto', 'background-color': '#313131', 'font-weight': 'bold', 'color': 'white', 'text-align': 'center'}),
        html.Button('Clear', style={'BackgroundColor':'white', 'font-size': 18, 'text-align': 'center'}, id='reset', n_clicks=0),
        html.Div(style={'columnCount': 2, 'background-color': 'rgb(251, 251, 233)'}, children = [
            dcc.Graph(id='scatter_plot_one'),
            dcc.Graph(id='scatter_plot_two'),
            dcc.Graph(id='bi_plot'),
            dcc.Graph(id='correlation_matrix'),
            dcc.Graph(id='bar_chart'),
            dcc.Graph(id='map')
        ])
])

# ----------------------------------------------------Callback and Other Functions---------------------------------------------

def plot(dataFrame, xVal, yVal, selectedIntersection, typeOfPlot):

    graph = ""
    if typeOfPlot == 'ScatterPlot':
        # Create a scatter plot.
        scatterPlot = px.scatter(
            dataFrame,
            x = xVal,
            y = yVal,
            size = yVal,
            color = 'State_ID',
            title = labelDictionary[xVal] + ' VS ' + labelDictionary[yVal],
            hover_data = {'State_ID'},
            labels = labelDictionary,
            color_continuous_scale = plotly.colors.sequential.Burg
        )

        scatterPlot.update_traces(selectedpoints=selectedIntersection, customdata=dataFrame.index, unselected_marker_opacity=0.05, unselected_marker_color='white')

        # Update layout of scatter plot.
        scatterPlot.update_layout(
            font = dict (
                size = 12
            ),
            title_x = 0.5,
            title_y = 0.90,
            width = 700,
            height = 600,
            dragmode = 'select',
            clickmode = 'event+select',
            showlegend=False,
            paper_bgcolor='rgb(251, 251, 233)',
            plot_bgcolor='black',
            coloraxis_showscale=False
        )

        graph = scatterPlot

    elif typeOfPlot == 'BarChart':

        barChart = px.bar(
            dataFrame,
            y = yVal,
            orientation = 'h',
            color = 'State_ID',
            labels = labelDictionary,
            color_continuous_scale = plotly.colors.sequential.Burg
        )

        barChart.update_traces(selectedpoints=selectedIntersection, customdata=dataFrame.index, unselected_marker_opacity=0.2, unselected_marker_color='white')

        # Update layout of scatter plot.
        barChart.update_layout(
            title_text = "Places",
            title_x = 0.55,
            title_y = 0.98,
            font = dict (
                size = 12
            ),
            width = 700,
            height = 1800,
            dragmode = 'select',
            clickmode = 'event+select',
            paper_bgcolor='rgb(251, 251, 233)',
            plot_bgcolor='black',
            coloraxis_showscale=False
        )

        graph = barChart

    elif typeOfPlot == 'BiPlot':
        # Create BiPlot.
        # Store 10 columns(attributes).
        df = dataFrame[['Area_Land', 'Population', 'Rent_Mean', 'Rent_GT_50', 'HI_Mean', 'HC_Mortgage_Mean', 'Home_Equity_Second_Mortgage', 'Debt', 'HS_Degree', 'Married']]
        pcaThree = PCA(n_components = 2)
        a = StandardScaler().fit_transform(df)
        componentsBiPlot = pcaThree.fit_transform(a)
        features = ['Area_Land', 'Population', 'Rent_Mean', 'Rent_GT_50', 'HI_Mean', 'HC_Mortgage_Mean', 'Home_Equity_Second_Mortgage', 'Debt', 'HS_Degree', 'Married']
        loadings = pcaThree.components_.T * np.sqrt(pcaThree.explained_variance_)
        labels = {str(i): f'PC {i+1}' for i in range(2)}
        labels['color'] = 'State ID'
        labels['size'] = 'Mental Health Rank'
        biPlot = px.scatter (
            componentsBiPlot,
            x = 0,
            y = 1,
            size = dataFrame['State_Mental_Health_Rank'],
            color = dataFrame['State_ID'],
            color_continuous_scale = plotly.colors.sequential.Burg,
            labels = labels
        )

        biPlot.update_layout (
            width = 700,
            height = 600,
            title = {
                'text': 'BiPlot',
                'font': dict (
                    size = 17
                ),
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            font = dict (
                size = 12
            ),
            dragmode = 'select',
            clickmode = 'event+select',
            paper_bgcolor='rgb(251, 251, 233)',
            plot_bgcolor='black',
            coloraxis_showscale=False
        )


        for i, feature in enumerate(features):
            biPlot.add_shape (
                type = 'line',
                line=dict(color="White",width=3),
                x0 = 0,
                y0 = 0,
                x1 = loadings[i, 0],
                y1 = loadings[i, 1],
            )
            biPlot.add_annotation (
                x = loadings[i, 0],
                y = loadings[i, 1],
                ax = 0,
                ay = 0,
                xanchor = "center",
                yanchor = "bottom",
                text = feature,
                font=dict(
                    color="white"
                )
            )

        biPlot.update_traces(selectedpoints=selectedIntersection, customdata=df.index, unselected_marker_opacity=0.05, unselected_marker_color='white')

        graph = biPlot

    elif typeOfPlot == 'CorrelationMatrix':
        # Store 10 columns(attributes).
        df = dataFrame[['Area_Land', 'Population', 'Rent_Mean', 'Rent_GT_50', 'HI_Mean', 'HC_Mortgage_Mean', 'Home_Equity_Second_Mortgage', 'Debt', 'HS_Degree', 'Married']]
        # Create a correlation matrix.
        matrix = df.corr()
        correlation = go.Figure (
            data = go.Heatmap (
                z = matrix,
                x = ['Area_Land', 'Population', 'Rent_Mean', 'Rent_GT_50', 'HI_Mean', 'HC_Mortgage_Mean', 'Home_Equity_Second_Mortgage', 'Debt', 'HS_Degree', 'Married'],
                y = ['Area_Land', 'Population', 'Rent_Mean', 'Rent_GT_50', 'HI_Mean', 'HC_Mortgage_Mean', 'Home_Equity_Second_Mortgage', 'Debt', 'HS_Degree', 'Married'],
                colorscale = plotly.colors.sequential.Burg
             )
        )

        correlation.update_traces(showscale=False)

        correlation.update_layout (
            width = 700,
            height = 600,
            title = {
                'text': 'CorrelationMatrix',
                'font': dict (
                    size = 17
                ),
                'y': 0.89,
                'x': 0.59,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            font = dict (
                size = 12
            ),
            paper_bgcolor='rgb(251, 251, 233)',
            dragmode = 'select',
            clickmode = 'event+select'
        )

        graph = correlation

    elif typeOfPlot == 'Map':
        map = go.Figure(data=go.Choropleth(
                locationmode='USA-states',
                locations=dataFrame['State_Code'],
                z=dataFrame['State_Unemployment_Rate'],
                colorscale = plotly.colors.sequential.Burg
        ))

        map.update_layout (
            title_text = "State Unemployment Rates",
            title_x = 0.5,
            title_y = 0.78,
            dragmode = 'select',
            clickmode = 'event+select',
            geo_scope='usa',
            width = 700,
            height = 600,
            paper_bgcolor='rgb(251, 251, 233)',
            geo_bgcolor="black"
        )

        map.update_traces(selectedpoints=selectedIntersection, customdata=dataFrame.index, selector=dict(type='choropleth'), showscale=False, unselected_marker_opacity=0.005)

        graph = map

    return graph

numClicks = []

# Callback composed of output and input.
@app.callback(
    [Output(component_id = 'scatter_plot_one', component_property = 'figure'),
     Output(component_id = 'scatter_plot_two', component_property = 'figure'),
     Output(component_id = 'bar_chart', component_property = 'figure'),
     Output(component_id = 'correlation_matrix', component_property = 'figure'),
     Output(component_id = 'bi_plot', component_property = 'figure'),
     Output(component_id = 'map', component_property = 'figure')
    ],
    [Input(component_id = 'reset', component_property = 'n_clicks')],
    [Input(component_id = 'scatter_plot_one', component_property = 'clickData')],
    [Input(component_id = 'scatter_plot_one', component_property = 'selectedData')],
    [Input(component_id = 'scatter_plot_two', component_property = 'clickData')],
    [Input(component_id = 'scatter_plot_two', component_property = 'selectedData')],
    [Input(component_id = 'bar_chart', component_property = 'clickData')],
    [Input(component_id = 'bar_chart', component_property = 'selectedData')],
    [Input(component_id = 'bi_plot', component_property = 'clickData')],
    [Input(component_id = 'bi_plot', component_property = 'selectedData')],
    [Input(component_id = 'map', component_property = 'clickData')],
    [Input(component_id = 'map', component_property = 'selectedData')]
)

# Function that updates all the graphs according
# to what is selected.
def update_graphs(clicks, scatterOne_C, scatterOne_S, scatterTwo_C, scatterTwo_S, barOne_C, barOne_S, biPlot_C, biPlot_S, map_C, map_S):

    lengthOfArr = len(numClicks)
    if clicks not in numClicks:
        numClicks.append(clicks)
    newLengthOfArr = len(numClicks)
    if lengthOfArr == newLengthOfArr:
        # Make a copy of the data so the original does not change in any way.
        dataCopy = dataFrame

        selected = dataCopy.index

        selectedBar = []
        for data in [barOne_C, barOne_S]:
            if data and data['points']:
                for x in data['points']:
                    if x['y'] not in selectedBar:
                        selectedBar.append(x['y'])

        resultTwo = []
        for element in selectedBar:
            for ind in dataFrame.index:
                if dataFrame['Place'][ind] == element:
                    resultTwo.append(ind)

        selected = np.intersect1d(selected, resultTwo)

        selectedMap = []
        for data in [map_C, map_S]:
            if data and data['points']:
                for x in data['points']:
                    if x['location'] not in selectedMap:
                        selectedMap.append(x['location'])

        if len(selectedMap) == 1:
            if len(selected) == 0:
                selected = dataCopy.index
            result = []
            for ind in dataFrame.index:
                if dataFrame['State_Code'][ind] == selectedMap[0]:
                    result.append(ind)

            selected = np.intersect1d(selected, result)

        if len(selected) == 0:
            selected = dataCopy.index
            for data in [scatterOne_C, scatterOne_S, scatterTwo_C, scatterTwo_S, barOne_S, barOne_C, biPlot_C, biPlot_S, map_C, map_S]:
                if data and data['points']:
                    selected = np.intersect1d(selected, [x['customdata'] for x in data['points']])

        else:
            for data in [scatterOne_C, scatterOne_S, scatterTwo_C, scatterTwo_S, biPlot_C, biPlot_S]:
                if data and data['points']:
                    selected = np.intersect1d(selected, [x['customdata'] for x in data['points']])


        arr = [plot(dataCopy, "HI_Mean", "Debt", selected, "ScatterPlot"),
               plot(dataCopy, "Married", "HC_Mean", selected, "ScatterPlot"),
               plot(dataCopy, "", "Place", selected, "BarChart"),
               plot(dataCopy, "", "", selected, "CorrelationMatrix"),
               plot(dataCopy, "", "", selected, "BiPlot"),
               plot(dataCopy, "", "", selected, "Map")]

    else:
        # Make a copy of the data so the original does not change in any way.
        dataCopy = dataFrame
        selected = dataCopy.index
        arr = [plot(dataCopy, "HI_Mean", "Debt", selected, "ScatterPlot"),
               plot(dataCopy, "Married", "HC_Mean", selected, "ScatterPlot"),
               plot(dataCopy, "", "Place", selected, "BarChart"),
               plot(dataCopy, "", "", selected, "CorrelationMatrix"),
               plot(dataCopy, "", "", selected, "BiPlot"),
               plot(dataCopy, "", "", selected, "Map")]

    return arr

@app.callback(
    [Output(component_id = 'scatter_plot_one', component_property = 'clickData'),
     Output(component_id = 'scatter_plot_two', component_property = 'clickData'),
     Output(component_id = 'bar_chart', component_property = 'clickData'),
     Output(component_id = 'correlation_matrix', component_property = 'clickData'),
     Output(component_id = 'bi_plot', component_property = 'clickData'),
     Output(component_id = 'map', component_property = 'clickData')
    ],
    [Input(component_id = 'reset', component_property = 'n_clicks')],
)
def resetClicks(clicks):
    return [None, None, None, None, None, None]

@app.callback(
    [Output(component_id = 'scatter_plot_one', component_property = 'selectedData'),
     Output(component_id = 'scatter_plot_two', component_property = 'selectedData'),
     Output(component_id = 'bar_chart', component_property = 'selectedData'),
     Output(component_id = 'correlation_matrix', component_property = 'selectedData'),
     Output(component_id = 'bi_plot', component_property = 'selectedData'),
     Output(component_id = 'map', component_property = 'selectedData')
    ],
    [Input(component_id = 'reset', component_property = 'n_clicks')],
)
def resetSelection(select):
    return [None, None, None, None, None, None]

if __name__ == '__main__':
   app.run_server(debug='true')
