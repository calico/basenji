#!/usr/bin/env python
from optparse import OptionParser
import collections
import functools
import pdb
import sys

import numpy as np
import pandas as pd
import zarr

from google.cloud import bigquery

import dash
import dash.dependencies as dd
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.graph_objs as go

'''
basenji_fetch_app.py

Run a Dash app to enable SAD queries.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sad_zarr_path>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='chrom_zarrs',
        default=False, action='store_true',
        help='Zarr files split by chromosome [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide SAD zarr')
    else:
        sad_zarr_path = args[0]

    #############################################
    # precursors

    print('Preparing data.', flush=True)

    chr_sad_zarr_open = {}

    if not options.chrom_zarrs:
        # open Zarr
        sad_zarr_open = zarr.open_group(sad_zarr_path, 'r')

        # with one file, hash to a fake chromosome
        chr_sad_zarr_open = {1: sad_zarr_open}

        # hash SNP ids to indexes
        snps = np.array(sad_zarr_open['snp'])
        snp_indexes = {}
        for i, snp_id in enumerate(snps):
            snp_indexes[snp_id] = (1, i)
        del snps

    else:
        snp_indexes = {}

        for ci in range(1,23):
            # open Zarr
            # sad_zarr_open = zarr.open_group('%s/%d.zarr' % (sad_zarr_path,ci), 'r')
            sad_zarr_open = zarr.open_group('%s/chr%d/sad_table.zarr' % (sad_zarr_path,ci), 'r')

            # with one file, hash to a fake chromosome
            chr_sad_zarr_open[ci] = sad_zarr_open

            # hash SNP ids to indexes
            snps = np.array(sad_zarr_open['snp'])
            for i, snp_id in enumerate(snps):
                snp_indexes[snp_id] = (ci, i)
            del snps

        # open chr1 Zarr for non-chr specific data
        sad_zarr_open = chr_sad_zarr_open[1]

    # easy access to target information
    target_ids = np.array(sad_zarr_open['target_ids'])
    target_labels = np.array(sad_zarr_open['target_labels'])

    # read SAD percentile indexes into memory
    sad_pct = np.array(sad_zarr_open['SAD_pct'])

    # read percentiles
    percentiles = np.around(sad_zarr_open['percentiles'], 3)
    percentiles = np.append(percentiles, percentiles[-1])

    # initialize BigQuery client
    client = bigquery.Client('seqnn-170614')

    print('Done.', flush=True)

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
                        {'label':'DNase', 'value':'DNASE'},
                        {'label':'H3K4me3', 'value':'HISTONE:H3K4me3'},
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
                dcc.Input(id='snp_id', value='rs2157719', type='text'),
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
                rows=[],
                columns=['SNP', 'Association', 'Score', 'ScoreQ', 'R2', 'Experiment', 'Description'],
                column_widths=[150, 125, 125, 125, 125, 200],
                editable=False,
                filterable=True,
                sortable=True,
                resizable=True,
                sortColumn='Association',
                row_selectable=True,
                selected_row_indices=[],
                max_rows_in_viewport=20
            )
        ])
    ])


    #############################################
    # callback helpers

    @memoized
    def query_ld(snp_id, population):
        """Query Google Genomics 1000 Genomes LD table for the given SNP."""
        bq_path = 'genomics-public-data.linkage_disequilibrium_1000G_phase_3'

        # construct query
        query = 'SELECT tname, corr'
        query += ' FROM `%s.super_pop_%s`' % (bq_path,population)
        query += ' WHERE qname = "%s"' % snp_id

        # run query
        print('Running a BigQuery!', file=sys.stderr)
        query_results = client.query(query)

        return query_results

    @memoized
    def read_sad(chrom, snp_i, verbose=True):
        """Read SAD scores from Zarr for the given SNP index."""
        if verbose:
            print('Reading SAD!', file=sys.stderr)

        # read SAD
        snp_sad = chr_sad_zarr_open[chrom]['SAD'][snp_i,:].astype('float64')

        # compute percentile indexes
        snp_sadq = []
        for ti in range(len(snp_sad)):
            snp_sadq.append(int(np.searchsorted(sad_pct[ti], snp_sad[ti])))

        return snp_sad, snp_sadq

    def snp_rows(snp_id, dataset, ld_r2=1., verbose=True):
        """Construct table rows for the given SNP id and its LD set
           in the given dataset."""
        rows = []

        # search for SNP
        chrom, snp_i = snp_indexes.get(snp_id, (None,None))
        if chrom is not None:
            # SAD
            snp_sad, snp_sadq = read_sad(chrom, snp_i)

            # round floats
            snp_sad = np.around(snp_sad,4)
            snp_assoc = np.around(snp_sad*ld_r2, 4)
            ld_r2_round = np.around(ld_r2, 4)

            # extract target scores and info
            for ti, tid in enumerate(target_ids):
                if dataset == 'All' or target_labels[ti].startswith(dataset):
                    rows.append({
                        'SNP': snp_id,
                        'Association': snp_assoc[ti],
                        'Score': snp_sad[ti],
                        'ScoreQ': percentiles[snp_sadq[ti]],
                        'R2': ld_r2_round,
                        'Experiment': tid,
                        'Description': target_labels[ti]})
        elif verbose:
            print('Cannot find %s in snp_indexes.' % snp_id)

        return rows

    def make_data_mask(dataset):
        """Make a mask across targets for the given dataset."""
        dataset_mask = []
        for ti, tid in enumerate(target_ids):
            if dataset == 'All':
                dataset_mask.append(True)
            else:
                dataset_mask.append(target_labels[ti].startswith(dataset))
        return np.array(dataset_mask, dtype='bool')

    def snp_scores(snp_id, dataset, ld_r2=1.):
        """Compute an array of scores for this SNP
           in the specified dataset."""

        dataset_mask = make_data_mask(dataset)

        scores = np.zeros(dataset_mask.sum(), dtype='float64')

        # search for SNP
        if snp_id in snp_indexes:
            chrom, snp_i = snp_indexes[snp_id]

            # read SAD
            snp_sad, _ = read_sad(chrom, snp_i)

            # filter datasets
            snp_sad = snp_sad[dataset_mask]

            # add
            scores += snp_sad*ld_r2

        return scores

    #############################################
    # callbacks

    @app.callback(
        dd.Output('table', 'rows'),
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
            query_results = query_ld(snp_id, population)

            for ld_snp, ld_corr in query_results:
                rows += snp_rows(ld_snp, dataset, ld_corr)

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
            query_results = query_ld(snp_id, population)

            for ld_snp, ld_corr in query_results:
                query_scores += snp_scores(ld_snp, dataset, ld_corr)

        # sort
        sorted_indexes = np.argsort(query_scores)

        # range
        ymax = np.abs(query_scores).max()
        ymax *= 1.2

        return {
            'data': [go.Scatter(
                x=np.arange(len(query_scores)),
                y=query_scores[sorted_indexes],
                text=target_ids[target_mask][sorted_indexes],
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
    app.run_server(debug=False, port=8000)


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
