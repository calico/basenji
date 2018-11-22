#!/usr/bin/env python
from optparse import OptionParser
import collections
import functools
import os
import pdb
import sys

import numpy as np
import pandas as pd
import h5py

from google.cloud import bigquery

import dash
import dash_table as dt
import dash.dependencies as dd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from basenji.sad5 import ChrSAD5

'''
basenji_fetch_app.py

Run a Dash app to enable SAD queries.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sad_hdf5_path>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='chrom_hdf5',
        default=False, action='store_true',
        help='HDF5 files split by chromosome [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide SAD HDF5')
    else:
        sad_h5_path = args[0]

    #############################################
    # precursors

    print('Preparing data...', end='', flush=True)
    sad5 = ChrSAD5(sad_h5_path, index_chr=True)
    print('DONE.', flush=True)

    #############################################
    # layout

    column_widths = [('SNP',150), ('Association',125),
                     ('Score',125), ('ScoreQ',125), ('R',125),
                     ('Experiment',125), ('Description',200)]
    scc = [{'if': {'column_id': cw[0]}, 'width':cw[1]} for cw in column_widths]

    app = dash.Dash(__name__)
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
                        {'label':'DNase', 'value':'DNASE'},
                        {'label':'H3K4me3', 'value':'CHIP:H3K4me3'},
                        {'label':'All', 'value':'All'}
                    ],
                    value='CAGE'
                )
            ], style={'width': '250', 'display': 'inline-block'}),

            html.Div([
                html.Label('Population'),
                dcc.Dropdown(
                    id='population',
                    options=[
                        {'label':'-', 'value':'-'},
                        {'label':'1kG African', 'value':'AFR'},
                        {'label':'1kG American', 'value':'AMR'},
                        {'label':'1kG East Asian', 'value':'EAS'},
                        {'label':'1kG European', 'value':'EUR'},
                        {'label':'1kG South Asian', 'value':'SAS'}
                    ],
                    value='-'
                )
            ], style={'width': '250', 'display': 'inline-block'}),

            html.Div([
                html.Label('SNP ID'),
                dcc.Input(id='snp_id', value='rs6656401', type='text'),
                html.Button(id='snp_submit', n_clicks=0, children='Submit')
            ], style={'display': 'inline-block', 'float': 'right'})

        ], style={
            'borderBottom': 'thin lightgrey solid',
            'backgroundColor': 'rgb(250, 250, 250)',
            'padding': '10px 5px'
        }),

        dcc.Graph(id='assoc_plot'),

        html.Div([
            dt.DataTable(
                id='table',
                data=[],
                columns=[{'id':cw[0],'name':cw[0]} for cw in column_widths],
                style_cell_conditional=scc,
                editable=False,
                filtering=True,
                sorting=True,
                n_fixed_rows=20
            )
        ])
    ])

        # html.Div([
        #     dt.DataTable(
        #         id='table',
        #         data=[],
        #         columns=[cw[0] for cw in column_widths],
        #         style_cell_conditional=scc,
        #         editable=False,
        #         filtering=True,
        #         sorting=True,
        #         n_fixed_rows=20
        #     )


    #############################################
    # callback helpers

    @memoized
    def query_ld(population, snp_id):
        try:
            sad5.set_population(population)

        except ValueError:
            print('Population unavailable.', file=sys.stderr)
            return pd.DataFrame()

        chrm, snp_i = sad5.snp_chr_index(snp_id)
        pos = sad5.snp_pos(snp_i, chrm)

        if chrm is None:
            return pd.DataFrame()
        else:
            return sad5.emerald_vcf.query_ld(snp_id, chrm, pos, ld_threshold=0.8)

    @memoized
    def read_sad(chrm, snp_i, verbose=True):
        """Read SAD scores from HDF5 for the given SNP index."""
        if verbose:
            print('Reading SAD!', file=sys.stderr)

        # read SAD
        snp_sad = sad5.chr_sad5[chrm][snp_i].astype('float64')

        # read percentiles
        snp_pct = sad5.chr_sad5[chrm].sad_pct(snp_sad)

        return snp_sad, snp_pct

    def snp_rows(snp_id, dataset, ld_r2=1., verbose=True):
        """Construct table rows for the given SNP id and its LD set
           in the given dataset."""
        rows = []

        # search for SNP
        # chrom, snp_i = snp_indexes.get(snp_id, (None,None))
        chrm, snp_i = sad5.snp_chr_index(snp_id)

        if chrm is not None:
            # SAD
            snp_sad, snp_pct = read_sad(chrm, snp_i)

            # round floats
            snp_sad = np.around(snp_sad,4)
            snp_assoc = np.around(snp_sad*ld_r2, 4)
            ld_r2_round = np.around(ld_r2, 4)

            # extract target scores and info
            for ti, tid in enumerate(sad5.target_ids):
                if dataset == 'All' or sad5.target_labels[ti].startswith(dataset):
                    rows.append({
                        'SNP': snp_id,
                        'Association': snp_assoc[ti],
                        'Score': snp_sad[ti],
                        'ScoreQ': snp_pct[ti],
                        'R': ld_r2_round,
                        'Experiment': tid,
                        'Description': sad5.target_labels[ti]})
        elif verbose:
            print('Cannot find %s in snp_indexes.' % snp_id)

        return rows

    def make_data_mask(dataset):
        """Make a mask across targets for the given dataset."""
        dataset_mask = []
        for ti, tid in enumerate(sad5.target_ids):
            if dataset == 'All':
                dataset_mask.append(True)
            else:
                dataset_mask.append(sad5.target_labels[ti].startswith(dataset))
        return np.array(dataset_mask, dtype='bool')

    def snp_scores(snp_id, dataset, ld_r2=1.):
        """Compute an array of scores for this SNP
           in the specified dataset."""

        dataset_mask = make_data_mask(dataset)

        scores = np.zeros(dataset_mask.sum(), dtype='float64')

        # search for SNP
        chrm, snp_i = sad5.snp_chr_index(snp_id)
        if snp_i is not None:
            # read SAD
            snp_sad, _ = read_sad(chrm, snp_i)

            # filter datasets
            snp_sad = snp_sad[dataset_mask]

            # add
            scores += snp_sad*ld_r2

        return scores

    #############################################
    # callbacks

    @app.callback(
        dd.Output('table', 'data'),
        [dd.Input('snp_submit', 'n_clicks')],
        [
            dd.State('snp_id','value'),
            dd.State('dataset','value'),
            dd.State('population','value')
        ]
    )
    def update_table(n_clicks, snp_id, dataset, population, verbose=True):
        """Update the table with a new parameter set."""
        if verbose:
            print('Tabling')

        # add snp_id rows
        rows = snp_rows(snp_id, dataset)

        if population != '-':
            df_ld = query_ld(population, snp_id)
            for i, v in df_ld.iterrows():
                rows += snp_rows(v.snp, dataset, v.r)

        return rows

    @app.callback(
        dd.Output('assoc_plot', 'figure'),
        [dd.Input('snp_submit', 'n_clicks')],
        [
            dd.State('snp_id','value'),
            dd.State('dataset','value'),
            dd.State('population','value')
        ]
    )
    def update_plot(n_clicks, snp_id, dataset, population, verbose=True):
        if verbose:
            print('Plotting')

        target_mask = make_data_mask(dataset)

        # add snp_id rows
        query_scores = snp_scores(snp_id, dataset)

        if population != '-':
            df_ld = query_ld(population, snp_id)
            for i, v in df_ld.iterrows():
                query_scores += snp_scores(v.snp, dataset, v.r)

        # sort
        sorted_indexes = np.argsort(query_scores)

        # range
        ymax = np.abs(query_scores).max()
        ymax *= 1.2

        return {
            'data': [go.Scatter(
                x=np.arange(len(query_scores)),
                y=query_scores[sorted_indexes],
                text=sad5.target_ids[target_mask][sorted_indexes],
                mode='markers'
            )],
            'layout': {
                'height': 400,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'yaxis': {'range': [-ymax,ymax]},
                'xaxis': {'range': [-1,1+len(query_scores)]}
            }
        }

    #############################################
    # run

    app.scripts.config.serve_locally = True
    app.run_server(debug=False, port=8787)


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
