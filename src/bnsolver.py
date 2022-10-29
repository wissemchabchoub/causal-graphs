import numpy as np
from itertools import *
from scipy.optimize import minimize
import networkx as nx
from src.dag import DAG
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show
import pprint
import networkx as nx
import hvplot.networkx as hvnx
import holoviews as hv

output_notebook()


class BN_solver():
    """
    A class to fit the BN parameters

    """

    def __init__(self, dag, random_state=42):
        """
        init method

        Parameters
        ----------
        dag : DAG
            fitted dag object
        random_state : int
            random state

        """
        self.random_state = random_state
        if dag.fitted:
            self.dag = dag
        else:
            raise Exception("The dag is not fitted")

    def fit(self):
        """
        Fits the Bayesian Network

        Parameters
        ----------
        dag : DAG
            fitted dag object
        random_state : int
            random state

        """
        self.edges = self.dag.convert_edges()
        self.theta_map = self.dag.create_theta_map()
        self.gamma_prime_map = self.dag.create_gamma_prime_map()

        self.observed_edges_proba = self.dag.observed_edges_proba
        self.directions = self.dag.directions
        self.edges_names = self.dag.edges
        self.nodes = self.dag.nodes

        self.V = len(self.nodes)
        self.H = len(self.theta_map) + 1
        self.JPT_config = np.array(list(product([0, 1], repeat=self.V)))

        self.solve()

        self.get_fitted()
        self.odds()
        self.dag.add_odds(self.odds)

    def predict(self, prior):
        """
        Infer marginal probability for all nodes

        """
        marginals = {}

        for node in self.nodes:
            marginals[node] = self.infer_marginals_node(
                self.dag.g, node, 1, prior)

        return marginals

    def plot_fitted_edge_proba(self):
        """
        Plots the fitted edge proba vs the observed edge proba

        """

        plt.rcParams["figure.figsize"] = (5, 5)
        plt.scatter(self.observed_edges_proba.flatten(),
                    np.array(list(self.fitted.values())).flatten())
        plt.xlim(right=1)
        plt.ylim(top=1)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.plot((0, 1), (0, 1), linestyle='-', color='k',
                 lw=3, scalex=False, scaley=False)
        plt.xlabel('observed')
        plt.ylabel('fitted')
        plt.show()

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(list(self.dag.g.edges(data=True)))

    def plot(self):
        """
        Plots the Bayesian Network
        """

        pos = nx.spring_layout(self.dag.g)

        odds_lst = np.array(list(self.odds.values()))
        odds_lst[odds_lst < 1] = 1 / odds_lst[odds_lst < 1]
        odds_lst = np.log(odds_lst)
        odds_rescaled = (odds_lst - odds_lst.min()) / \
            (odds_lst.max() - odds_lst.min()) + 0.05

        obj = hvnx.draw(self.dag.g, pos, edge_width=odds_rescaled*10, arrowhead_length=0.1,
                        node_size=3000, with_labels=True, node_color='grey',)

        return obj

    def plot_predict(self, prior):
        """
        Plots the Bayesian Network with Inferred Marginal Probas  
        """

        predictions = self.predict(prior)
        self.dag.add_infer_marg(predictions)

        pos = nx.spring_layout(self.dag.g_with_proba)
        odds_lst = np.array(list(self.odds.values()))
        odds_lst[odds_lst < 1] = 1 / odds_lst[odds_lst < 1]
        odds_lst = np.log(odds_lst)
        odds_rescaled = (odds_lst - odds_lst.min()) / \
            (odds_lst.max() - odds_lst.min()) + 0.05

        obj = hvnx.draw(self.dag.g_with_proba, pos, edge_width=odds_rescaled*10, arrowhead_length=0.1,
                        node_size=3000, with_labels=True, node_color='grey',)

        return obj

    def prior_format(self):
        """
        Prints the prior input format 
        """
        print('Nodes :')
        print(self.dag.g.nodes)
        print([['1-{}_proba_up'.format(node), '{}_proba_up'.format(node)]
              if self.is_root(node) else [0, 0] for node in self.dag.g.nodes])

    def build_observations(self):
        """
        Creates the observations dict

        Parameters
        ----------

        Returns
        -------
        dict
            observations dict
        """

        observations = []
        for i, edge in enumerate(self.edges):
            for j, direction in enumerate(self.directions):
                observation = {'indexes': edge, 'values': direction,
                               'proba': self.observed_edges_proba[i, j]}
                observations.append(observation)

        return observations

    def build_X(self):
        """
        Creates X

        Parameters
        ----------

        Returns
        -------
        X : array
            X
        """

        X = np.zeros((2 ** self.V, self.H))
        for i, config in enumerate(self.JPT_config):
            x = [1]
            for j in range(self.H - 1):
                x.append(np.prod(
                    np.where(config[self.theta_map[j]] == 0, -1, config[self.theta_map[j]])))
            x = np.array(x)
            X[i] = x
        return X

    def build_M(self, observations):
        """
        Creates M for each observation

        Parameters
        ----------
        observations : dict
            dict of observations

        Returns
        -------
        observations : dict
            dict of observations
        """

        for observation in observations:
            M = np.zeros(2 ** self.V)
            for i in range(2 ** self.V):
                if np.all(self.JPT_config[i][observation['indexes']] == observation['values']):
                    M[i] = 1
            observation['M'] = M
        return observations

    def build_A(self):
        """
        Creates A

        Parameters
        ----------

        Returns
        -------
        array
            A
        """

        self.T = int(len(self.gamma_prime_map) / 2)
        A = np.zeros((2 ** self.V, 2 * self.T))
        for i, gamma_desciption in enumerate(self.gamma_prime_map):
            for j, config in enumerate(self.JPT_config):
                if np.all(config[gamma_desciption['indexes']] == gamma_desciption['values']):
                    A[:, i][j] = 1
        return A

    def solve(self):
        """
        Calculates the parameters of the BN

        Parameters
        ----------

        Returns
        -------
        array
            Joint probability table
        """

        observations = self.build_observations()
        observations = self.build_M(observations)
        self.observations = observations
        X = self.build_X()
        A = self.build_A()

        def funct(params):
            JPT_LL = np.exp(X.dot(params[:self.H])) / \
                np.sum(np.exp(X.dot(params[:self.H])))
            JPT_BN = np.exp(A.dot(np.log(params[self.H:])))

            # Loss
            loss = 0
            for observation in observations:
                loss += (observation['proba'] -
                         JPT_LL.dot(observation['M'])) ** 2

            loss += np.sum((JPT_BN - JPT_LL) ** 2)

            return loss

        abstol = 1e-8

        cons = ({'type': 'ineq', 'fun': lambda x: x[self.H:] - abstol})

        rng = np.random.RandomState(self.random_state)

        res = minimize(funct, x0=rng.uniform(size=self.H + 2 * self.T), constraints=cons,
                       options={'maxiter': 500})

        self.theta = res.x[:self.H]
        self.gamma_prime = res.x[self.H:]
        self.JPT = np.exp(A.dot(np.log(res.x[self.H:])))

        return self.JPT

    def get_fitted(self):
        """
        Calculates the fitted observed edge probas

        Parameters
        ----------

        Returns
        -------
        dict
            fitted observed edge probas for each edge
        """

        fitted = []
        for i in range(len(self.observations)):
            fitted.append(self.JPT.dot(self.observations[i]['M']))

        fitted = np.array(fitted).reshape(self.observed_edges_proba.shape)

        self.fitted = {}

        for i in range(len(fitted)):
            self.fitted[str(self.edges_names[i])] = fitted[i]

    def odds(self):
        """
        Calculates the odds for each edge

        Parameters
        ----------

        Returns
        -------
        dict
            odds for each edge
        """

        fitted = []
        for i in range(len(self.observations)):
            fitted.append(self.JPT.dot(self.observations[i]['M']))

        fitted = np.array(fitted).reshape(self.observed_edges_proba.shape)

        odds = {}
        for i in range(len(fitted)):
            odds[str(self.edges_names[i])] = (fitted[i][0] *
                                              fitted[i][3]) / (fitted[i][1] * fitted[i][2])
        self.odds = odds

    def marginals(self):
        """
        Calculates the marginal proba for each node

        Parameters
        ----------

        Returns
        -------
        dict
            marginal proba for each node
        """

        marg = {}
        for i, node in enumerate(self.nodes):
            P = 0
            for j in range(len(self.JPT_config)):
                P += self.JPT[j] * self.JPT_config[j, i]
            marg[node] = P
        return marg

    def is_root(self, node):
        """
        Verifies if a node is root

        """
        return list(self.dag.g.predecessors(node)) == []

    def build_marginals_array(self):
        """
        Builds an len_nodes by 2 array of BN marginals

        """

        marginals = []

        for i, node in enumerate(self.dag.g.nodes):

            P = 0

            for j in range(len(self.JPT_config)):
                P += self.JPT[j] * self.JPT_config[j, i]

            marginals.append([1 - P, P])

        return np.array(marginals)

    def marginalize(self, nodes, values):
        """
        Computes marginal probability of list of nodes        
        """

        P = 0

        for i, config in enumerate(self.JPT_config):
            if np.all(config[nodes] == values):
                P += self.JPT[i]
        return P

    def infer_marginals_node(self, graph, node, value, prior=None):
        """
        Infer marginal probability of a node

        """

        if prior is None:
            prior = self.prior

        # marginals = self.build_marginals_array(graph)

        ancestors = nx.ancestors(graph, node)

        roots = [anc for anc in ancestors if self.is_root(anc)]

        roots_index = [list(graph.nodes()).index(p) for p in roots]

        node_index = list(graph.nodes()).index(node)

        if not roots:
            return prior[node_index][value]

        P = 0
        for i in range(len(self.JPT_config)):

            if self.JPT_config[i][node_index] == value:
                P += self.JPT[i] * np.product(prior[roots_index, self.JPT_config[i][roots_index]]) / self.marginalize(
                    roots_index, self.JPT_config[i][roots_index])

                # np.product(marginals[roots_index,self.JPT_config[i][roots_index]])

        return P
