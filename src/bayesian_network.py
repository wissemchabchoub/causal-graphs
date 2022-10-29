from src.dag import DAG
from src.bnsolver import BN_solver


class BayesianNetwork():

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, data, threshold, date_start, date_end):
        self.dag = DAG(data, threshold, date_start=date_start,
                       date_end=date_end)
        self.dag.fit()
        self.bn = BN_solver(self.dag, self.random_state)
        self.bn.fit()

    def plot_fitted_edge_proba(self):
        self.bn.plot_fitted_edge_proba()

    def prior_format(self):
        self.bn.prior_format()

    def predict(self, prior):
        return self.bn.predict(prior=prior)

    def plot(self):
        return self.bn.plot()

    def plot_predict(self, prior):
        return self.bn.plot_predict(prior)
