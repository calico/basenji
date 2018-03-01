#!/usr/bin/env python
from optparse import OptionParser
import collections
import functools
import pdb

import numpy as np
import pandas as pd
import zarr

from google.cloud import bigquery

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
    # parser.add_option('--ld', dest='ld_query', default=False, action='store_true')
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

    # initialize BigQuery client
    client = bigquery.Client('seqnn-170614')

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
                        {'label':'-', 'value':'-'},
                        {'label':'1kG African', 'value':'1kG_AFR'},
                        {'label':'1kG American', 'value':'1kG_AMR'},
                        {'label':'1kG East Asian', 'value':'1kg_EAS'},
                        {'label':'1kG European', 'value':'1kg_EUR'},
                        {'label':'1kG South Asian', 'value':'1kg_SAS'}
                    ],
                    value='-'
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
                columns=['SNP', 'Association', 'Score', 'R2', 'Experiment', 'Description'],
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
    # callback helpers

    @memoized
    def read_sad(snp_i):
        return sad_zarr_in['sad'][snp_i,:].astype('float64')

    def snp_rows(input_snp, ld_r2=1.):
        rows = []

        # search for SNP
        if input_snp in snp_indexes:
            snp_i = snp_indexes[input_snp]
            snp_sad = np.around(read_sad(snp_i),4)
            snp_assoc = np.around(snp_sad*ld_r2, 4)

            # extract target scores and info
            for ti, tid in enumerate(target_ids):
                rows.append({
                    'SNP': input_snp,
                    'Association': snp_assoc[ti],
                    'Score': snp_sad[ti],
                    'R2': ld_r2,
                    'Experiment': tid,
                    'Description': target_labels[ti]})

        return rows

    #############################################
    # callbacks

    @app.callback(
        dd.Output('table', 'rows'),
        [dd.Input('snp_submit', 'n_clicks')],
        [dd.State('snp_id', 'value'), dd.State('population', 'value')]
    )
    def update_table(n_clicks, input_snp, population):
        # add input_snp rows
        rows = snp_rows(input_snp)

        if population != '-':
            # construct LD query
            query = 'SELECT tname, corr'
            query += ' FROM `genomics-public-data.linkage_disequilibrium_1000G_phase_3.super_pop_EAS`'
            query += ' WHERE qname = "%s"' % input_snp

            # query to DataFrame
            query_results = client.query(query)

            for ld_snp, ld_corr in query_results:
                rows += snp_rows(ld_snp, ld_corr)

        return rows


    #############################################
    # run

    app.scripts.config.serve_locally = True
    app.run_server(debug=True)


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
