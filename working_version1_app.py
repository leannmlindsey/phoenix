from dash import Dash, html, dcc, Output, Input, callback_context
import plotly.express as px
import pandas as pd

#########################################################################################
"""
app.py

Author: [Your Name]
Contributors: [List of Co-Authors or Collaborators]

Description:
This Dash application is part of the Phoenix prophage detection framework. It visualizes prophage detection results using an interactive heatmap. Users can select different datasets representing reference genomes, zoom into specific genome locations, and reset the view as needed.

Usage:
- Select a reference genome from the dropdown menu to load the corresponding dataset.
- Use the slider at the bottom of the heatmap to zoom into specific genome regions.
- Click the 'Reset Zoom' button to reset the view to the full genome.

Citation:
If you use this software or data in your research, please cite:

[Insert full citation here]
[For example:]
Doe, J., Smith, A., & Johnson, L. (2023). Phoenix Prophage Detection Framework: A Tool for Visualizing Prophage Genomic Data. *Journal of Computational Biology*, 30(4), 123-134. doi:10.1234/jcb.2023.5678

Acknowledgments:
This application was developed with contributions from the Phoenix research team at [Your Institution or Organization]. Special thanks to all team members for their valuable input and support.

License:
[Specify the license under which your code is distributed, e.g., MIT License, GPL, etc.]
"""
#########################################################################################
# Load the sample heatmap data
df = pd.read_csv('data/test2.csv')
df.columns = pd.to_numeric(df.columns, errors='coerce')
df.index = ['gLM Raw signal', 'PHASTER', 'REFERENCE']
##########################################################################################
# Create the heatmap figure
fig = px.imshow(
	df, 
	color_continuous_scale=px.colors.sequential.Blues,
	title="Prophage Detection Heatmap"
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
    tickfont=dict(size=16)    # Reduce y-axis tick font size
)

fig.update_traces(
    hovertemplate='genome location: %{x}<br>phage prediction from: %{y}<extra></extra>'
)
#############################################################################################
image_path = 'assets/phoenixlogo.jpg'
app = Dash(__name__)

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
                    html.P("3. Use the mouse to select the area of interest and zoom in. Controls on the top right provide the ability to download the image as a png, reset the axis (click on the house icon)."),
                ],
                style={'margin-left': '30px', 'textAlign': 'left'}
            )
        ],
        style={'display': 'flex', 'align-items': 'center'}
    ),

    # Button to reset zoom
    #html.Button('Reset Zoom', id='reset-zoom-button', n_clicks=0, style={
    #    'position': 'absolute',
    #    'top': '10px',
    #    'right': '10px',
    #    'zIndex': '999'
    #}),

    # Heatmap graph component
    dcc.Graph(id='graph-content', figure=fig, config={'displayModeBar': True})
],style={'position':'relative'})

# Callback to reset the zoom
#@app.callback(
#    Output('graph-content', 'figure'),
#    Input('reset-zoom-button', 'n_clicks'),
#    prevent_initial_call=True
#)

#def reset_zoom(n_clicks):
#    # Update the figure's x-axis range to the original range
#    fig.update_xaxes(range=original_x_range)
#    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
