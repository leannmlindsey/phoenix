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
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import label
import sys
print(sys.version)

# Read the accession file
accession_df = pd.read_csv('data/Accessions.csv', sep=',')
accession_df = accession_df.drop_duplicates(subset=['Assembly_short']).sort_values('Organism Name')
accession_df = accession_df.dropna(subset=['phylum'])
# Prepare options for the dropdown
dropdown_options = [
    {'label': f"{row['Assembly_short']} - {row['Organism Name']}", 'value': row['Assembly_short']}
    for index, row in accession_df.iterrows()
]

dataset_filter_options = ['Casjens','Phoenix','DEPHT','Phaster']

taxonomic_levels = ['kingdom','phylum','class','order','family','genus','species']
initial_taxon = 'phylum'
taxonomic_filter = accession_df[initial_taxon].unique()
initial_taxon_filter = 'No Filter'
#algorithm_options = [{'label': 'Moving Window Sum', 'value': 'mws'},
#                    {'label': 'Moving Window Average', 'value': 'mwa'},
#                    {'label': 'Median Filter', 'value': 'median'},
#                    {'label': 'Run Length Encoding', 'value': 'rle'},
#                    {'label': 'dbScan', 'value': 'dbscan'},
#                   {'label': 'Connected Component Labeling', 'value': 'ccl'}]

algorithm_options = [{'label': 'Moving Window Sum', 'value': 'mws'},
                    {'label': 'Moving Window Average', 'value': 'mwa'},
                    {'label': 'Median Filter', 'value': 'median'},
                    ]
# Set the image path
image_path = 'assets/phoenixlogo.jpg'

# Initialize the app
app = Dash(__name__)
server = app.server  # Expose the Flask server instance

# Define initial accession
initial_accession = accession_df['Assembly_short'].iloc[0]
initial_dataset = 'Phoenix'
initial_algorithm = 'mws'

colors = {
    'yellow': 'rgba(249, 199, 13, 0.7)',
    'blue': 'rgba(78, 120, 166, 0.7)',
    'dk_blue': 'rgba(78, 120, 166, 1.0)',
    'white': 'rgba(255, 255, 255, 0.7)',
    'green': 'rgba(18, 168, 138, 0.7)',
    'rose': 'rgba(203, 71, 121, 0.7)',
    'terracotta': 'rgba(219, 112, 87, 0.7)',
    'purple': 'rgba(123, 97, 166, 0.7)',
    'teal': 'rgba(56, 163, 165, 0.7)',
    'gold': 'rgba(212, 175, 55, 0.7)',
    'sage': 'rgba(144, 169, 85, 0.7)',
    'mauve': 'rgba(171, 142, 154, 0.7)',
    'black': 'rgba(0, 0, 0, 0.7)'
}

threshold_defaults = {
    'mws': 0.4,
    'mwa': 0.2,  
    'rle': 0.7,  
    'dbscan': 0.3,
    'median': 0.5,
    'ccl': 0.4
}
# Load initial dataset
csv_path = os.path.join('data/combined', f'processed_{initial_accession}_combined.csv')

app.layout = html.Div([
    # Main container with flex display
    html.Div([
        # Left Panel
        html.Div([
            # Logo
            html.Img(
                src=image_path,
                style={
                    'width': '100%',
                    'max-width': '300px',
                    'margin-bottom': '20px'
                }
            ),

            # Controls Section
            html.Div([
                html.Div([
                    html.Label('Dataset:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Dropdown(
                        id='dataset-filter-dropdown',
                        options=dataset_filter_options,
                        value=initial_dataset,
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'margin-bottom': '20px'}),
                html.Div([
                    html.Label('Taxonomic Level:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Dropdown(
                        id='taxonomic_levels',
                        options=taxonomic_levels,
                        value=initial_taxon,
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'margin-bottom': '20px'}),

                html.Div([
                    html.Label('Taxon Filter:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Dropdown(
                        id='taxonomic_filter',
                        options=taxonomic_filter,
                        value=initial_taxon_filter,
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'margin-bottom': '20px'}),
                # Dataset Dropdown
                html.Div([
                    html.Label('Genome:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=dropdown_options,
                        value=initial_accession,
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'margin-bottom': '20px'}),

                html.Div([
                    html.Label('Algorithm:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Dropdown(
                        id='algorithm',
                        options=algorithm_options,
                        value=initial_algorithm,
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'margin-bottom': '20px'}),

                # Threshold Input
                html.Div([
                    html.Label('Threshold Value:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Input(
                        id='threshold-input',
                        type='number',
                        value=threshold_defaults['mwa'],
                        min=0,
                        step=0.01,
                        style={'width': '100%', 'padding': '5px'}
                    )
                ], style={'margin-bottom': '20px'}),
                
                #html.Div([
                #    html.Label('Genomic Language Model:', style={'font-weight': 'bold', 'margin-bottom': '2%'}),
                #    dcc.Checklist(
                #        id='gLM',
                #        options=[
                #            #{'label': 'CNN', 'value': 'cnn'},
                #            {'label': 'DNABERT2', 'value': 'dnabert2'},
                #            {'label': 'Caduceus', 'value': 'caduceus'},
                #            #{'label': 'Nucleotide Transformer', 'value': 'nt'},
                #            {'label': 'Grover', 'value': 'grover'},
                #            #{'label': 'Mistral-DNA', 'value': 'mistral-dna'},
                #            #{'label': 'GENA-LM', 'value': 'gena-lm'},
                           
                #        ],
                #        value=['DNABERT2'],  # default value
                #        labelStyle={'display': 'block'}
                #    )
                #], style={'margin-bottom': '5%'}),

                

                # Space for future controls
                html.Div(id='future-controls', style={'margin-bottom': '20px'}),

                
            ], style={'padding': '20px'})
        ], style={
            'width': '400px',  
            'background-color': '#f8f9fa',  # Light gray background
            'padding': '20px',
            'border-right': '1px solid #dee2e6',  # Border on the right
            'height': '100vh',  # Full height
            'overflow-y': 'auto'  # Scrollable if content is too long
        }),
        # Main Content Area (Graph)
        html.Div([
            dcc.Graph(
                id='graph-content',
                config={'displayModeBar': True},
                style={
                    'height': 'calc(100vh - 40px)',  # Full height minus padding
                    'width': '100%'
                }
            )
        ], style={
            'flex': '1',  # Takes up remaining space
            'padding': '20px',
            'overflow': 'hidden'  # Prevents scrolling
        })
    ], style={
        'display': 'flex',
        'height': '100vh',
        'width': '100%'
    })
], style={
    'margin': '0',
    'padding': '0',
    'height': '100vh'
})

# Function to create the plot based on selected data
def create_heatmap_figure(dataset_filter, taxon_filter, taxon_list, accession, threshold_value=25, algorithm='mws'):
    try:
        # Construct the path to the dataset CSV file
        csv_path = os.path.join('data/combined', f'processed_{accession}_combined.csv')

        # Check if the file exists
        if not os.path.exists(csv_path):
            # Handle the case where the file does not exist
            # Create a figure with an error message
            fig = px.imshow([[0]], text_auto=True)
            fig.update_layout(
                title="Error loading data",
                xaxis_visible=False,
                yaxis_visible=False,
                annotations=[dict(text=f"Dataset file {csv_path} not found.", showarrow=False,
                                  xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))]
            )
            return fig

        # Load the dataset
        df = pd.read_csv(csv_path)
        
        df.rename(columns={
            'caduceus_pred': 'Predictions',
            'dnabert2_pred': 'DNABERT2',
            'grover_pred': 'Grover',
            'reference_label': 'Reference',
            'median': 'Median',
            'mwa': 'Moving Weight Average',
            'rle': 'Run Length Encoding',
            'dbscan': 'DBScan',
            'ccl': 'Connected Component Labeling',
            'window_sum': 'Window Summation'
        }, inplace=True)
        
        colors = {
            'yellow': 'rgba(249, 199, 13, 0.7)',
            'blue': 'rgba(78, 120, 166, 0.7)',
            'dk_blue': 'rgba(78, 120, 166, 1.0)',
            'white': 'rgba(255, 255, 255, 0.7)',
            'green': 'rgba(18, 168, 138, 0.7)',
            'rose': 'rgba(203, 71, 121, 0.7)',
            'terracotta': 'rgba(219, 112, 87, 0.7)',
            'purple': 'rgba(123, 97, 166, 0.7)',
            'teal': 'rgba(56, 163, 165, 0.7)',
            'gold': 'rgba(212, 175, 55, 0.7)',
            'sage': 'rgba(144, 169, 85, 0.7)',
            'mauve': 'rgba(171, 142, 154, 0.7)',
            'black': 'rgba(0, 0, 0, 0.7)'
        }

        # add Scoring Algorithm
        if algorithm == 'mws':
            df['algo'] = df['Predictions'].rolling(window=85,center=True, min_periods=1).sum()/85
            df['algo'] = df['algo'].fillna(0)
            df['predicted_interval'] = [1 if x > threshold_value else 0 for x in df['algo']]

        elif algorithm == 'mwa':   # Moving Window Average
            # Calculate moving window average
            df['algo'] = df['Predictions'].rolling(window=70,center=True, min_periods=1).mean()
            df['algo'] = df['algo'].fillna(0)
            df['predicted_interval'] = [1 if x > threshold_value else 0 for x in df['algo']]
            
        elif algorithm == 'rle':  # Run Length Encoding
           df['algo'] = df['Run Length Encoding']
           df['predicted_interval'] = df['algo']


        elif algorithm == 'dbscan':  # DBScan
           df['algo'] = df['DBScan']
           df['predicted_interval'] = df['algo']


        elif algorithm == 'median':  # Median Filter
           # Apply median filtering
           df['algo'] = df['Predictions'].rolling(window=60,center=True, min_periods=1).median()
           df['algo'] = df['algo'].fillna(0)
           df['predicted_interval'] = [1 if x > threshold_value else 0 for x in df['algo']]


        elif algorithm == 'ccl':  # Connected Component Labeling
            df['algo'] = df['Connected Component Labeling']
            df['predicted_interval'] = df['algo']

        fig = plotly.subplots.make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.2,0.2,0.05,0.5,0.15,0.15,0.15])

        # Plot score data on the 1st subplot
        fig.add_trace(go.Scatter(x=df.index, y=df["Reference"],
                          fill='tonexty',
                          name='Reference Locations',
                          line_color=colors['yellow'],
                          fillcolor=colors['yellow'],
                          legendrank=5
                          ))

        # Plot predicted phage signal on 2nd row
        fig.add_trace(go.Scatter(x=df.index, y=df["predicted_interval"],
                 fill='tonexty',
                 name='gLM Predicted Locations',
                 line_color=colors['rose'],
                 fillcolor=colors['rose'],
                 legendrank=4
              ), row=2, col=1)
        

        # Plot GC on 3th row
        fig.add_trace(go.Scatter(x=df.index, y=df["gc"],
                 fill='tonexty',
                 name='Moving Window Sum',
                 line_color=colors['yellow'],
                 fillcolor=colors['yellow'],
                 legendrank=3
              ), row=3, col=1)

        # Plot Moving Window Sum on 4nd row
        fig.add_trace(go.Scatter(x=df.index, y=df["algo"],
                 fill='tonexty',
                 name='Moving Window Sum',
                 line_color=colors['dk_blue'],
                 fillcolor=colors['dk_blue'],
                 legendrank=2
              ), row=4, col=1)
        fig.add_hline(y=threshold_value, line_dash="dash", line_color="black", row=4, col=1,
              annotation_text="Threshold", 
              annotation_position="right")
        

        # Plot raw gLM signal on 5th row
        fig.add_trace(go.Scatter(x=df.index,
                 y=df['Predictions'], 
                 fill='tozeroy',
                 name='Caduceus signal',
                 line_color=colors['dk_blue'],
                 fillcolor=colors['dk_blue'],
                 legendrank=1
                 ), row=5, col=1)

        # Plot DNABERT2 signal on 6th row
        fig.add_trace(go.Scatter(
                x=df.index,
                y=df['DNABERT2'],
                fill='tozeroy',
                name='DNABERT2 signal',
                line_color=colors['terracotta'], 
                fillcolor=colors['terracotta'],
                legendrank=0
        ), row=6, col=1)

        # Plot Grover signal on 7th row
        fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Grover'],
                fill='tozeroy',
                name='Grover signal',
                line_color=colors['teal'],  
                fillcolor=colors['teal'],
                legendrank=-1
        ), row=7, col=1)

        fig.update_yaxes(title_text="Reference",title_standoff=0,showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Predicted",title_standoff=0,showticklabels=False, row=2, col=1)
        fig.update_yaxes(title_text="GC",title_standoff=0,showticklabels=False, range=[df["gc"].min(), df["gc"].max()], row=3, col=1)
        fig.update_yaxes(title_text="Score",title_standoff=0,showticklabels=False, row=4, col=1)
        fig.update_yaxes(title_text="Caduceus",title_standoff=0,showticklabels=False, row=5, col=1)
        fig.update_yaxes(title_text="DNABERT2", title_standoff=0, showticklabels=False, row=6, col=1)  
        fig.update_yaxes(title_text="Grover", title_standoff=0, showticklabels=False, row=7, col=1)  
        fig.update_yaxes(dtick=10)  
        fig.update_xaxes(dtick=100000)  

        fig.update_layout(
            height=1000,  # Fixed height in pixels
            autosize=True,  # Allow the figure to resize horizontally
            showlegend=True,
            xaxis_rangeslider_visible=False,
            xaxis7_rangeslider_visible=True,
            xaxis7=dict(
                rangeslider=dict(
                    thickness=0.05,
                )
            ),
            margin=dict(l=50, r=50, t=30, b=30)  # Fixed margins in pixels
        )


        fig.update_layout(
            legend=dict(
                x=1,  # x-coordinate of the legend (0 is left, 1 is right)
                y=1,  # y-coordinate of the legend (0 is bottom, 1 is top)
                xanchor="right",  # anchor point of the legend on the x-axis
                yanchor="top",  # anchor point of the legend on the y-axis
            )       
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

# Callback to update the heatmap when the dataset or selected rows change
# Callback to update all components
@app.callback(
    [Output('taxonomic_filter', 'options'),
     Output('taxonomic_filter', 'value'),
     Output('dataset-dropdown', 'options'),
     Output('dataset-dropdown', 'value'),
     Output('graph-content', 'figure')],
    [Input('dataset-filter-dropdown', 'value'),
     Input('taxonomic_levels', 'value'),
     Input('taxonomic_filter', 'value'),
     Input('dataset-dropdown', 'value'),
     Input('threshold-input', 'value'),
     Input('algorithm', 'value')]
)
def update_dashboard(dataset_filter, taxon_level, taxon_filter, selected_accession, threshold_value, algorithm):
    # First filter by dataset
    filtered_df = accession_df[accession_df[dataset_filter] == 1]
    
    # Get unique values for the selected taxonomic level
    taxon_options = sorted(filtered_df[taxon_level].unique())
    
    # Create taxonomic filter options with "No Filter" as the first option
    taxon_dropdown_options = [{'label': 'No Filter', 'value': 'no_filter'}] + [
        {'label': str(taxon), 'value': str(taxon)} for taxon in taxon_options
    ]
    
    # Set default to "No Filter" if no filter is selected or if current filter is invalid
    if not taxon_filter or str(taxon_filter) not in [opt['value'] for opt in taxon_dropdown_options]:
        taxon_filter = 'no_filter'
    
    # Apply taxonomic filter only if a specific taxon is selected
    df_for_genome_list = filtered_df
    if taxon_filter != 'no_filter' and taxon_level and taxon_filter:
        df_for_genome_list = filtered_df[filtered_df[taxon_level].astype(str) == str(taxon_filter)]
    
    # Create genome dropdown options from filtered dataframe
    genome_options = [
        {'label': f"{row['Assembly_short']} - {row['Organism Name']}", 'value': row['Assembly_short']}
        for index, row in df_for_genome_list.iterrows()
    ]
    
    # If current genome selection is not in new options, select the first available option
    if not genome_options or selected_accession not in [opt['value'] for opt in genome_options]:
        selected_accession = df_for_genome_list['Assembly_short'].iloc[0] if not df_for_genome_list.empty else None
    
    # Get the list of all taxa for the selected level
    taxon_list = df_for_genome_list[taxon_level].unique() if not df_for_genome_list.empty else []
    
    # Create the figure
    fig = create_heatmap_figure(dataset_filter, taxon_filter, taxon_list, selected_accession, threshold_value, algorithm)
    
    return (
        taxon_dropdown_options,
        taxon_filter,
        genome_options,
        selected_accession,
        fig
    )
if __name__ == '__main__':
    app.run_server(debug=True)

