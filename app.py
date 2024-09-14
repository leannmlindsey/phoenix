"""
app.py

Author: LeAnn M. Lindsey
Contributors: Nicole L. Pershing, Anisa Habib, Aaron Schindler, June Round, W. Zac Stephens, Anne J. Blaschke, and Hari Sundar

Description:
This Dash application is part of the Phoenix prophage detection framework. It visualizes prophage detection results using an interactive heatmap. Users can select different datasets representing reference genomes, zoom into specific genome locations, and reset the view as needed.

Usage:
- Select a reference genome from the dropdown menu to load the corresponding dataset.
- Use the slider at the bottom of the heatmap to zoom into specific genome regions.
- Use the mouse to select the area of interest and zoom in. Controls on the top right provide the ability to download the image as a PNG, reset the axis (click on the house icon).

Citation:
If you use this software or data in your research, please cite:

Lindsey, L. et al., (2024) Phoenix: A Prophage Signal Detection Framework for Genomic Language Models.

License:
This code is distributed under the MIT License.
"""

from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import os

# Read the accession file
accession_df = pd.read_csv('data/accession.csv')

# Prepare options for the dropdown
dropdown_options = [{'label': accession, 'value': accession} for accession in accession_df['Accession']]

# Set the image path
image_path = 'assets/phoenixlogo.jpg'

# Initialize the app
app = Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.Div(
        [
            # Logo image on the left
            html.Img(src=image_path, style={'height': '300px', 'margin-left': '50px'}),

            # Instructions on the right
            html.Div(
                [
                    html.H3("Instructions"),
                    html.P("1. Select a reference genome from the dropdown menu."),
                    html.P("2. The initial view shown is the full genome. The colored bars are the locations of prophage in the reference, predicted by PHASTER and predicted by the gLM."),
                    html.P("3. Use the mouse to select the area of interest and zoom in. Controls on the top right provide the ability to download the image as a PNG, reset the axis (click on the house icon)."),
                    html.P("If you use this software or data in your research, please cite: Lindsey L., Pershing N.L., Habib A., Schindler A., Round J., Stephens W.Z., Blaschke A.J., and Sundar H. (2024) Phoenix: A Prophage Signal Detection Framework for Genomic Language Models."),
                    html.P("Build a dashboard with your own data using instructions from www.github.com/leannmlindsey/Phoenix.git")
                ],
                style={'margin-left': '30px', 'textAlign': 'left'}
            )
        ],
        style={'display': 'flex', 'align-items': 'center'}
    ),

    # Dropdown to select the dataset
    html.Div(
        [
            html.Label('Select Dataset:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=dropdown_options,
                value=accession_df['Accession'].iloc[0],  # Default value
                clearable=False,
                style={'width': '300px'}
            )
        ],
        style={'margin-left': '50px', 'margin-top': '20px'}
    ),

    # Heatmap graph component
    dcc.Graph(id='graph-content', config={'displayModeBar': True})
], style={'position': 'relative'})

# Function to create the heatmap figure based on the selected dataset
def create_heatmap_figure(accession):
    try:
        # Construct the path to the dataset CSV file
        csv_path = os.path.join('data', f'{accession}.csv')

        # Check if the file exists
        if not os.path.exists(csv_path):
            # Handle the case where the file does not exist
            print(f"Dataset file {csv_path} not found.")
            # Create a figure with an error message
            fig = px.imshow([[0]], text_auto=True)
            fig.update_layout(
                title="Error Loading Data",
                xaxis_visible=False,
                yaxis_visible=False,
                annotations=[dict(text=f"Dataset file {csv_path} not found.", showarrow=False,
                                  xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))]
            )
            return fig

        # Load the dataset
        df = pd.read_csv(csv_path)
        df.columns = pd.to_numeric(df.columns, errors='coerce')
        df.index = ['gLM Raw signal', 'PHASTER', 'REFERENCE']

        # Create the heatmap figure
        fig = px.imshow(
            df,
            color_continuous_scale=px.colors.sequential.Blues,
            title=f"Prophage Detection Heatmap - {accession}"
        )

        # Update x-axis rangeslider visibility
        fig.update_xaxes(rangeslider_visible=True)

        # Adjust the layout to make the heatmap taller and adjust fonts
        fig.update_layout(
            hovermode='x unified',
            height=800,  # Increase figure height to make the heatmap taller
        )

        # Adjust x-axis tick labels
        fig.update_xaxes(
            tickfont=dict(size=8),   # Reduce x-axis tick font size
            tickangle=45,            # Rotate x-axis labels if they overlap
            tickformat="~s",         # Format large numbers in SI notation (e.g., 1M for 1,000,000)
        )

        # Adjust y-axis tick labels
        fig.update_yaxes(
            tickfont=dict(size=16)    # Adjust y-axis tick font size
        )

        # Customize the hover information
        fig.update_traces(
            hovertemplate='genome location: %{x}<br>phage prediction from: %{y}<extra></extra>'
        )

        return fig

    except Exception as e:
        # If there is an error, return an empty figure or display an error message
        print(f"Error loading data for {accession}: {e}")
        # Create a figure with an error message
        fig = px.imshow([[0]], text_auto=True)
        fig.update_layout(
            title="Error Loading Data",
            xaxis_visible=False,
            yaxis_visible=False,
            annotations=[dict(text="Error loading data. Please check the data file.", showarrow=False,
                              xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))]
        )
        return fig

# Callback to update the heatmap when a new dataset is selected
@app.callback(
    Output('graph-content', 'figure'),
    Input('dataset-dropdown', 'value')
)
def update_heatmap(selected_accession):
    # Create the figure
    fig = create_heatmap_figure(selected_accession)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

