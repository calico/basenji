#!/usr/bin/env python
from optparse import OptionParser
import pdb

import numpy as np
import zarr

import dash
import dash.dependencies as dd
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

'''
basenji_fetch_app.py

Run a Dash app to enable SAD queries.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sad_zarr_file>'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide SAD zarr')
    else:
        sad_zarr_file = args[0]

    #############################################
    # precursors

    sad_zarr_in = zarr.open_group(sad_zarr_file)

    # hash SNP ids to indexes
    snp_indexes = {}
    for i, snp_id in enumerate(sad_zarr_in['snps']):
        snp_indexes[snp_id] = i

    # easy access to target information
    target_ids = sad_zarr_in['target_ids']
    target_labels = sad_zarr_in['target_labels']
    sad = sad_zarr_in['sad']


    #############################################
    # layout

    app = dash.Dash()
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

    app.layout = html.Div([
        html.Div([
            html.H1('Basenji SNP activity difference'),

            dcc.Markdown('Instructions...'),

            html.Div([
                html.Label('Datasets'),
                dcc.Dropdown(
                    id='dataset',
                    options=[
                        {'label':'CAGE', 'value':'CAGE'},
                        {'label':'DNase', 'value':'DNase'},
                        {'label':'H3K4me3', 'value':'H3K4me3'}
                    ],
                    value='DNase'
                )
            ], style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                html.Label('Population'),
                dcc.Dropdown(
                    id='population',
                    options=[
                        {'label':'1kG European', 'value':'1kg_euro'},
                        {'label':'1kG Japanese', 'value':'1kg_jap'}
                    ],
                    value='1kg_euro'
                )
            ], style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                html.Label('SNP ID'),
                dcc.Input(id='snp_id', value='rs2157719', type='text'),
                html.Button(id='snp_submit', n_clicks=0, children='Submit')
            ], style={'display': 'inline-block', 'float': 'right'})

        ], style={
            'borderBottom': 'thin lightgrey solid',
            'backgroundColor': 'rgb(250, 250, 250)',
            'padding': '10px 5px'
        }),

        html.Div([
            dt.DataTable(
                id='table',
                rows=[],
                columns=['SNP', 'Score', 'Experiment', 'Description'],
                column_widths=[200],
                editable=False,
                filterable=True,
                sortable=True,
                resizable=True,
                sortColumn='Score',
                row_selectable=True,
                selected_row_indices=[],
                max_rows_in_viewport=20
            )
        ])
    ])


    #############################################
    # callbacks

    @app.callback(
        dd.Output('table', 'rows'),
        [dd.Input('snp_submit', 'n_clicks')],
        [dd.State('snp_id', 'value')]
    )
    def update_table(n_clicks, input_snp):
        # search for SNP
        if snp_id not in snp_indexes:
            raise ValueError('%s not found')
        else:
            snp_i = snp_indexes[input_snp]
            snp_sad = np.around(sad[snp_i,:].astype('float64'),4)

            # extract target scores and info
            rows = []
            for ti, tid in enumerate(target_ids):
                rows.append({
                    'SNP': input_snp,
                    'Score': snp_sad[ti],
                    'Experiment': tid,
                    'Description': target_labels[ti]})

            return rows


    #############################################
    # run

    app.scripts.config.serve_locally = True
    app.run_server(debug=True)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
