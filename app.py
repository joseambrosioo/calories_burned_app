import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the data
# Replace with the actual path to your CSV files
try:
    calories = pd.read_csv('calories.csv')
    exercise_data = pd.read_csv('exercise.csv')
    # Combine the dataframes
    calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
    # Convert 'Gender' to numerical for analysis
    calories_data['Gender'] = calories_data['Gender'].replace({'Male': 0, 'Female': 1})
except FileNotFoundError:
    print("Error: The CSV files were not found. Please check the file paths.")
    calories_data = pd.DataFrame() # Create an empty dataframe to avoid errors

# Define the models and their performance metrics including MSE and RMSE
models = {
    'XGBoost': {'mae': 3.49, 'r2': 0.9995, 'mse': 19.33, 'rmse': 4.39},
    'Random Forest': {'mae': 2.75, 'r2': 0.9997, 'mse': 10.96, 'rmse': 3.31},
    'Gradient Boosting': {'mae': 4.19, 'r2': 0.9992, 'mse': 26.65, 'rmse': 5.16},
    'Extra Trees': {'mae': 2.22, 'r2': 0.9998, 'mse': 7.64, 'rmse': 2.76},
    'KNeighbors': {'mae': 15.06, 'r2': 0.9705, 'mse': 549.99, 'rmse': 23.45},
    'Linear Regression': {'mae': 7.14, 'r2': 0.9945, 'mse': 156.4, 'rmse': 12.51},
    'Lasso': {'mae': 7.14, 'r2': 0.9945, 'mse': 156.4, 'rmse': 12.51},
    'LGBM': {'mae': 3.84, 'r2': 0.9994, 'mse': 17.55, 'rmse': 4.19},
    'Ridge': {'mae': 7.14, 'r2': 0.9945, 'mse': 156.4, 'rmse': 12.51},
    'SVR': {'mae': 8.91, 'r2': 0.9912, 'mse': 252.32, 'rmse': 15.88},
    'ElasticNet': {'mae': 12.6, 'r2': 0.9814, 'mse': 531.02, 'rmse': 23.04},
    'Decision Tree': {'mae': 4.35, 'r2': 0.9992, 'mse': 25.13, 'rmse': 5.01},
    'Huber': {'mae': 8.16, 'r2': 0.9922, 'mse': 222.92, 'rmse': 14.93},
    'Bayesian Ridge': {'mae': 7.14, 'r2': 0.9945, 'mse': 156.4, 'rmse': 12.51},
    'MLP': {'mae': 20.35, 'r2': 0.9507, 'mse': 1419.03, 'rmse': 37.67},
    'NNR': {'mae': 2.12, 'r2': 0.9998, 'mse': 7.82, 'rmse': 2.79}
}

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Calorie Burned Prediction Dashboard"
server = app.server

# Define the app layout
app.layout = html.Div(
    style={'font-family': 'sans-serif', 'padding': '20px'},
    children=[
        html.H1("🏋️ Calorie Burned Prediction Dashboard 🏃", style={'text-align': 'center'}),
        html.P(
            "Welcome! This dashboard explores the relationship between various exercise and physiological "
            "metrics and the number of calories burned. Use the interactive tools below to explore the data and "
            "compare the performance of different machine learning models.",
            style={'text-align': 'justify', 'margin-bottom': '20px'}
        ),

        html.Hr(),

        ## Data Exploration Section
        html.H2("📊 Data Exploration"),
        html.P(
            "Select a variable to visualize its distribution and relationship with other features."
        ),
        html.Div([
            html.Label("Select a feature to visualize:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in calories_data.columns if col not in ['User_ID']],
                value='Duration'
            )
        ], style={'width': '50%', 'margin-bottom': '20px'}),

        dcc.Graph(id='dist-graph'),
        dcc.Graph(id='scatter-graph'),

        html.Hr(),

        ## Model Comparison Section
        html.H2("🧠 Model Performance"),
        html.P(
            "This section compares different machine learning models used to predict calories burned. "
            "Lower values for MAE, MSE, and RMSE indicate better performance, while a higher R-squared score is better."
        ),
        html.P(
            "The **Mean Squared Error (MSE)** is the average of the squared errors, giving more weight to larger errors. "
            "The **Root Mean Squared Error (RMSE)** is the square root of the MSE, which is useful because the units are the same as the target variable (Calories).",
            style={'text-align': 'justify'}
        ),
        html.P(
            "The **Mean Absolute Error (MAE)** measures the average magnitude of the errors, and the **R-squared Score (R²)** measures the proportion of variance explained by the model."
        ),

        html.Div([
            html.H3("Model Performance Table", style={'margin-top': '20px'}),
            html.Table(
                id='model-table',
                style={'width': '100%', 'border-collapse': 'collapse'}
            )
        ], style={'margin-bottom': '20px'}),

        dcc.Graph(id='mae-graph'),
        dcc.Graph(id='r2-graph'),
        dcc.Graph(id='mse-graph'),
        dcc.Graph(id='rmse-graph'),
    ]
)

# Callback to update the distribution and scatter graphs
@app.callback(
    [Output('dist-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('feature-dropdown', 'value')]
)
def update_graphs(selected_feature):
    if calories_data.empty:
        return {}, {}
    # Distribution Plot
    dist_fig = px.histogram(calories_data, x=selected_feature, marginal='box',
                            title=f'Distribution of {selected_feature}')
    dist_fig.update_layout(bargap=0.1)

    # Scatter Plot vs. Calories
    scatter_fig = px.scatter(calories_data, x=selected_feature, y='Calories', color='Gender',
                             title=f'{selected_feature} vs. Calories Burned')
    return dist_fig, scatter_fig

# Callback to update the model performance table and graphs
@app.callback(
    [Output('model-table', 'children'),
     Output('mae-graph', 'figure'),
     Output('r2-graph', 'figure'),
     Output('mse-graph', 'figure'),
     Output('rmse-graph', 'figure')],
    [Input('model-table', 'id')]
)
def update_model_info(_):
    # Create table header
    header = [
        html.Tr([
            html.Th("Model", style={'border': '1px solid black', 'padding': '8px', 'background-color': '#f2f2f2'}),
            html.Th("MAE", style={'border': '1px solid black', 'padding': '8px', 'background-color': '#f2f2f2'}),
            html.Th("R²", style={'border': '1px solid black', 'padding': '8px', 'background-color': '#f2f2f2'}),
            html.Th("MSE", style={'border': '1px solid black', 'padding': '8px', 'background-color': '#f2f2f2'}),
            html.Th("RMSE", style={'border': '1px solid black', 'padding': '8px', 'background-color': '#f2f2f2'})
        ])
    ]
    # Create table rows
    rows = [
        html.Tr([
            html.Td(model_name, style={'border': '1px solid black', 'padding': '8px'}),
            html.Td(f"{metrics['mae']:.2f}", style={'border': '1px solid black', 'padding': '8px'}),
            html.Td(f"{metrics['r2']:.4f}", style={'border': '1px solid black', 'padding': '8px'}),
            html.Td(f"{metrics['mse']:.2f}", style={'border': '1px solid black', 'padding': '8px'}),
            html.Td(f"{metrics['rmse']:.2f}", style={'border': '1px solid black', 'padding': '8px'})
        ]) for model_name, metrics in models.items()
    ]
    table = header + rows

    # Create bar charts for each metric
    mae_fig = px.bar(
        x=list(models.keys()),
        y=[m['mae'] for m in models.values()],
        labels={'x': 'Model', 'y': 'Mean Absolute Error'},
        title='MAE for Each Model'
    )
    r2_fig = px.bar(
        x=list(models.keys()),
        y=[m['r2'] for m in models.values()],
        labels={'x': 'Model', 'y': 'R-squared Score'},
        title='R-squared Score for Each Model'
    )
    mse_fig = px.bar(
        x=list(models.keys()),
        y=[m['mse'] for m in models.values()],
        labels={'x': 'Model', 'y': 'Mean Squared Error'},
        title='MSE for Each Model'
    )
    rmse_fig = px.bar(
        x=list(models.keys()),
        y=[m['rmse'] for m in models.values()],
        labels={'x': 'Model', 'y': 'Root Mean Squared Error'},
        title='RMSE for Each Model'
    )

    # Update layouts to rotate x-axis labels
    for fig in [mae_fig, r2_fig, mse_fig, rmse_fig]:
        fig.update_layout(xaxis_tickangle=-45)

    return table, mae_fig, r2_fig, mse_fig, rmse_fig


server = app.server # This exposes the Flask server object to Gunicorn


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)