import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from itertools import *
import pandas as pd


class DAG():
    """
    A class to create a DAG using causal data

    """

    def __init__(self, data, threshold, date_start, date_end):
        """
        init method

        Parameters
        ----------
        data : dataframe
            data
        directions : list
            list of directions of the observed probabilities
        threshold : int
            threshold to add an edge
        date_start : date
            start date
        date_end : date
            end date

        """

        self.date_start = pd.to_datetime(date_start, utc=True)
        self.date_end = pd.to_datetime(date_end, utc=True)
        self.directions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.threshold = threshold
        self.data = data
        self.fitted = False

    def fit(self):
        """
        Creates the dag
        """
        self.get_jpt()
        self.create_dag_from_data()
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        for edge in self.edges:
            g.add_edge(edge[0], edge[1], odds_ratio=0)
        self.g = g
        self.fitted = True

    def plot(self):
        """
        Plots the graph
        """
        plt.tight_layout()
        nx.draw_networkx(self.g, arrows=True)
        plt.show()
        plt.clf()

    def convert_edges(self):
        """
        Returns the edges indexes

        Returns
        -------
        list
            list of edges
        """
        edges_index = []
        for edge in self.edges:
            edges_index.append(
                [self.nodes.index(edge[0]), self.nodes.index(edge[1])])
        return edges_index

    def create_theta_map(self):
        """
        Returns the theta mapping

        Returns
        -------
        list
            theta map
        """
        nodes = list(self.nodes)
        theta_map = []

        for i, node in enumerate(nodes):
            theta_map.append([i])

        for node in self.g.pred.keys():
            if self.g.pred[node] != {}:
                theta_map += [[nodes.index(node), nodes.index(parent)]
                              for parent in self.g.pred[node].keys()]

        return theta_map

    def create_gamma_prime_map(self):
        """
        Returns the gamma_prime mapping

        Returns
        -------
        list
            gamma_prime map
        """
        nodes = list(self.g.nodes)
        gamma_1 = []
        gamma_2 = []

        for node in self.g.pred.keys():
            if self.g.pred[node] == {}:
                gamma_1.append(
                    {
                        'indexes': [nodes.index(node)],
                        'values': [1]
                    })

                gamma_2.append(
                    {
                        'indexes': [nodes.index(node)],
                        'values': [0]
                    })
            else:
                n_parents = len(self.g.pred[node])
                combs = product([0, 1], repeat=n_parents)

                for comb in combs:
                    gamma_1.append(
                        {
                            'indexes': [nodes.index(node)] + [nodes.index(parent) for parent in
                                                              self.g.pred[node].keys()],
                            'values': [1] + list(comb)
                        })

                    gamma_2.append(
                        {
                            'indexes': [nodes.index(node)] + [nodes.index(parent) for parent in
                                                              self.g.pred[node].keys()],
                            'values': [0] + list(comb)
                        })

        return gamma_1 + gamma_2

    def add_odds(self, odds):
        """
        Adds odds ratios to the graph edges

        Parameters
        ----------
        odds : dict
            odds for every edge
        """

        for i, edge in enumerate(self.edges):
            self.g[edge[0]][edge[1]]['odds_ratio'] = list(odds.values())[i]

    def add_infer_marg(self, infer_marg):
        """
        Adds probas to the graph nodes 

        Parameters
        ----------
        infer_marg : dict
            proba for every node
        """

        mapping = {node: node + '\n' +
                   str(int(proba * 100)) + ' %' for node, proba in infer_marg.items()}

        self.g_with_proba = nx.relabel_nodes(self.g, mapping)

    def plot_proba(self):
        """
        Plots the graph with proba
        """
        plt.tight_layout()
        nx.draw_networkx(self.g_with_proba, arrows=True)
        plt.show()
        plt.clf()

    def create_dag_from_data(self):
        """
        Creates the dag from jpt and occurences dataframe

        Parameters
        ----------
        data : dataframe
            data
        threshold : int
            threshold to add an edge
        Returns
        -------
        nx.DiGraph
            DAG
        """

        self.data.occurrences[self.data.occurrences <= self.threshold] = None
        self.data.dropna(inplace=True)
        self.data = self.data.sort_values('occurrences', ascending=False)

        g = nx.DiGraph()

        for _, row in self.data.iterrows():
            g0 = g.copy()
            g.add_edge(row['cause'], row['effect'])
            if not nx.is_directed_acyclic_graph(g):
                g = g0.copy()

        observed_edges_proba = []
        for edge in g.edges:
            observed_edges_proba.append(
                self.data[np.logical_and(self.data.cause == edge[0], self.data.effect == edge[1])].proba.values[0])

        g.remove_nodes_from(list(nx.isolates(g)))

        self.edges = list(g.edges)
        self.nodes = list(g.nodes)
        self.observed_edges_proba = np.array(
            observed_edges_proba).reshape((len(g.edges), 4))

    def get_jpt(self):
        """
        Creates a JPT dataframe from data dataframe

        Parameters
        ----------
        data : dataframe
            data
        date_start : date
            start date
        date_end : date
            end date
        Returns
        -------
        nx.DiGraph
            DAG
        """
        self.data = self.data[np.logical_and(
            self.data.date >= self.date_start, self.data.date <= self.date_end)]

        self.data = self.data.groupby(['cause', 'effect'], as_index=False).apply(lambda x: pd.Series({'occurrences': x['sentence'].count(),
                                                                                                      'proba': self.directions_to_proba(x['directions'], self.directions)}))

    @staticmethod
    def directions_to_proba(directions, directions_map):
        """
        Helper function for the JPT dataframe
        """
        P = np.array([sum(directions == d) for d in directions_map]) + 0.5
        return P / sum(P)
