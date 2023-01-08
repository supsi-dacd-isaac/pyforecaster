import numpy as np
import pandas as pd
from jax import numpy as jnp, jit, vmap, grad
from functools import partial
from typing import Tuple, Union
from pyforecaster.scenred import scenred, retrieve_scenarios_indexes, plot_from_graph, plot_vars, replace_var, \
    get_nodes_per_time_from_tree, plot_graph
from abc import abstractmethod
import matplotlib.pyplot as plt
from os.path import join
import networkx as nx


class NeuralScenarioTree:
    def __init__(self, tree=None, nodes_at_step=None, savepath='', init='quantiles', base_tree='scenred'):
        self.nodes_at_step = nodes_at_step
        self.tree = tree
        self.cm = plt.get_cmap('viridis', 10)
        self.savepath = savepath
        self.init = init
        self.base_tree = base_tree

    @abstractmethod
    def gen_tree(self, scens:Union[list, np.ndarray, pd.DataFrame], start_tree, k_max=1000, tol=1e-3):
        if start_tree is None:
            tree = self.gen_init_tree(scens)
        tree_idxs, leaves = retrieve_scenarios_indexes(tree)
        tree_vals = np.hstack(list(dict(tree.nodes('v')).values()))
        tree_scens = np.vstack([tree_vals[idx] for idx in tree_idxs])
        return tree, tree_scens, tree_idxs, tree_vals

    def init_vals(self, tree, tree_scens, tree_vals, scens):
        if self.init == 'zero':
            tree_scens *= 0
            tree_vals *= 0
            replace_var(tree, tree_vals)
        elif self.init == 'quantiles':
            bins = {0: np.quantile(scens[0, :], [0, 1])}
            vals = {0: np.median(scens[0, :])}
            filters = {0: np.arange(scens.shape[1])}
            for t in range(scens.shape[0]-1):
                nodes_at_t = [k for k,v in nx.get_node_attributes(tree, 't').items() if v == t]
                for j in nodes_at_t:
                    # compute quantiles of current node's children
                    children = list(tree.successors(j))
                    qs = np.quantile(scens[t+1, filters[j]], np.linspace(0, 1, len(children) + 1))
                    bins.update({c: [qs[i], qs[i+1]] for i, c in enumerate(children)})
                    vals_j = np.quantile(scens[t, filters[j]], np.linspace(0, 1, len(children) + 2)[1:-1])
                    for c, v in zip(children, vals_j):
                        vals[c] = v
                        filters[c] = filters[j][(scens[t+1, filters[j]] > bins[c][0] - 1e-6) &
                                                (scens[t+1, filters[j]] <= bins[c][1])]

            tree_idxs, leaves = retrieve_scenarios_indexes(tree)
            tree_scens = np.vstack([tree_vals[idx] for idx in tree_idxs])
            nx.set_node_attributes(tree, vals, name='v')
            tree_vals = np.hstack(list(dict(tree.nodes('v')).values()))
        return tree, tree_scens, tree_vals

    def gen_init_tree(self, scens):
        t, n_scens = scens.shape
        geometric_steps = np.array([2**t for t in range(int(np.log(scens.shape[1])/np.log(2)))][2:])
        geometric_progression = np.floor(np.logspace(-1, np.log(len(geometric_steps))/np.log(10), scens.shape[0])).astype(int)
        reverse_geom_progression = np.max(geometric_progression) - geometric_progression[::-1]
        geometric_nodes = geometric_steps[reverse_geom_progression]

        nodes_at_step = self.nodes_at_step if self.nodes_at_step is not None else \
            geometric_nodes

        if nodes_at_step[0] != 1:
            nodes_at_step[0] = 1
        if self.base_tree == 'scenred':
            _, _, _, _, tree = scenred(scens, nodes=nodes_at_step)
        elif self.base_tree == 'quantiles':
            # build valueless tree
            tree = nx.DiGraph()
            tree.add_node(0, t=0, p=1, v=np.atleast_1d(0))
            k = 1
            names_of_nodes_at_previous_step = [0]
            for t, n_t in enumerate(nodes_at_step):
                if t == 0:
                    continue
                names_of_nodes_at_t = []
                child_per_par = n_t / nodes_at_step[t-1]
                assert child_per_par == np.floor(child_per_par),  'children must be multiple of parent or the same'
                child_per_par = int(child_per_par)
                for p in names_of_nodes_at_previous_step:
                    for c in range(child_per_par):
                        names_of_nodes_at_t.append(k)
                        tree.add_node(k, t=t, p=1/n_t, v=np.atleast_1d(0))
                        tree.add_edge(p, k)
                        k += 1
                names_of_nodes_at_previous_step = np.copy(names_of_nodes_at_t)
            """
            qs = {t: np.quantile(scens[t, :], np.linspace(0, 1, n + 2)[1:-1]) for t, n in enumerate(nodes_at_step)}
            bins = {t: np.quantile(scens[t, :], np.linspace(0, 1, n + 1)) for t, n in enumerate(nodes_at_step)}
            for k, v in bins.items():
                bin_vals = bins[k]
                bin_vals[0] -=1e-6
                bin_vals[-1] += 1e-6
                bins[k] = bin_vals
            # build connectionless tree
            tree = nx.DiGraph()
            k = 0
            for t, n in enumerate(nodes_at_step):
                for qj in qs[t]:
                    tree.add_node(k, t=t, p=1/n, v=np.atleast_1d(qj))
                    k += 1

            for t in range(1, len(nodes_at_step)):
                nodes_at_current_t = [n for n, time in nx.get_node_attributes(tree, 't').items() if time == t]
                nodes_at_previous_t = [n for n, time in nx.get_node_attributes(tree, 't').items() if time == t - 1]
                # parents choose children
                for j in range(len(qs[t-1])):
                    # scenarios in current quantile at previous time
                    scens_filt_tj = (scens[t-1, :] > bins[t-1][j] - (j == 0) * 1e-6) & (scens[t-1, :] <= bins[t-1][j + 1])
                    # to which quantile they'll go
                    children_list = np.argsort(np.bincount(np.digitize(scens[t, scens_filt_tj], bins[t])))[1:]-1
                    for i in children_list[::-1]:
                        child = nodes_at_current_t[i]
                        if len(list(tree.predecessors(child))) <1:
                            tree.add_edge(nodes_at_previous_t[j], child)
                            break

                # children choose parents
                for j in range(len(qs[t])):
                    child = nodes_at_current_t[j]
                    # scenarios in current quantile at current time
                    scens_filt_tj = (scens[t, :] > bins[t][j] - (j==0)*1e-6) & (scens[t, :] <= bins[t][j+1])
                    # from which quantile they come from
                    rel_parent = np.argmax(np.bincount(np.digitize(scens[t-1, scens_filt_tj], bins[t-1])-1))
                    parent = nodes_at_previous_t[rel_parent]
                    if len(list(tree.predecessors(child))) == 0:
                        tree.add_edge(parent, child)

        """
        leaves = [n for n, time in nx.get_node_attributes(tree, 't').items() if time == len(nodes_at_step)-1]
        leaves_prob = {l: 1/len(leaves) for l in leaves}
        nx.set_node_attributes(tree, leaves_prob, name='p')
        nodes_prob = {}
        for n in list(set(tree.nodes)-set(leaves)):
            nodes_prob[n] = len([i for i in nx.descendants(tree, n) if i not in leaves]) / len(leaves)
        nx.set_node_attributes(tree, nodes_prob, name='p')
        # plot_graph(tree)
        return tree

    @staticmethod
    @jit
    @partial(vmap, in_axes=(1, None))
    def compute_distances(scens, x):
        return jnp.mean((x-scens)**2)

    @staticmethod
    @jit
    def metric_loss(tree_vals, tree_idxs, scens):
        tot_dist = jnp.array(0)
        tree_scens = jnp.vstack([tree_vals[i] for i in tree_idxs.T]).T
        for scen in scens.T:
            dists = NeuralScenarioTree.compute_distances(tree_scens, scen)
            tot_dist += jnp.min(dists)
        return tot_dist


class NeuralGas(NeuralScenarioTree):
    def __init__(self, tree=None, nodes_at_step=None, savepath='', init='quantiles', base_tree='quantiles'):
        self.pars = {'lambda_0': 5,
                     'lambda_f': 0.05,
                     'e0': 5,
                     'ef': 0.05}
        super().__init__(tree, nodes_at_step, savepath, init, base_tree)

    def gen_tree(self, scens: Union[list, np.ndarray, pd.DataFrame], start_tree=None, k_max=10000, tol=1e-3, do_plot=True):
        scens = np.array(scens)
        tree, tree_scens, tree_idxs, tree_vals = super().gen_tree(scens, start_tree)
        tree, tree_scens, tree_vals = self.init_vals(tree, tree_scens, tree_vals, scens)
        k = 0
        rel_dev = 1
        if do_plot:
            fig, ax = plt.subplots(1, 1)
        while rel_dev > tol and k < k_max:
            if k%1==0:
                tree_vals = update_tree_from_scenarios(tree, tree_idxs, tree_scens)
                loss = self.metric_loss(jnp.array(tree_vals).ravel(), jnp.array(tree_idxs), jnp.array(scens))
                print('iter {}, loss: {}'.format(k, loss))
            if do_plot and k%1 == 0:
                ax.cla()
                ax.plot(tree_scens, color=self.cm(2))
                ax.plot(scens, alpha=0.15, color=self.cm(5))
                ax.set_xlim(0, scens.shape[0]-1)
                plt.title('loss: {:0.3}'.format(loss))
                plt.savefig(join(self.savepath, 'step_{:03d}'.format(k)))
                plt.pause(0.01)
            # draw random realization
            scen = jnp.ravel(scens[:, np.random.choice(scens.shape[1], 1)])
            # compute distances and ranks from tree scenarios
            dists = self.compute_distances(tree_scens, scen)
            ranks = np.argsort(dists)
            # update pars
            e_k = self.pars['e0'] * (self.pars['ef'] / self.pars['e0']) ** (k / k_max)
            lambda_k = self.pars['lambda_0'] * (self.pars['lambda_f'] / self.pars['lambda_0']) ** (k / k_max)
            # modify scenario tree matrix through modified gradient descent
            for i, s in enumerate(tree_scens.T):
                err = scen - s
                tree_scens[:, i] += jnp.minimum(e_k * jnp.exp(-ranks[i] / lambda_k), 1) * err

            # reconcile scenario tree through averaging
            for t in range(tree_idxs.shape[0]):
                nodes_at_t = np.unique(tree_idxs[t, :])
                for u in nodes_at_t:
                    scen_filt = tree_idxs[t, :] == u
                    tree_scens[t, scen_filt] = jnp.mean(tree_scens[t, scen_filt])
            k += 1

        update_tree_from_scenarios(tree, tree_idxs, tree_scens)

        return tree


def update_tree_from_scenarios(tree, tree_idxs, tree_scens):
    tree_vals = []
    for i in range(len(tree.nodes)):
        var_pos = np.atleast_2d(np.argwhere(tree_idxs == i))
        var = np.unique([tree_scens[p[0], p[1]] for p in var_pos])
        if len(var) != 1:
            print('asda')
        assert len(var) == 1, 'smth wrong, var should contain just one obs (all obs in tree_scens at var_pos ' \
                              'should be equal by construction)'
        tree_vals.append(var)
    replace_var(tree, tree_vals)
    return tree_vals


class NeuralDiffTree(NeuralScenarioTree):
    def __init__(self, tree=None, nodes_at_step=None, savepath='', init='quantiles', base_tree='scenred'):
        super().__init__(tree, nodes_at_step, savepath, init, base_tree)

    def gen_tree(self, scens: Union[list, np.ndarray, pd.DataFrame], start_tree=None, k_max=100, tol=1e-3, do_plot=True):
        scens = np.array(scens)
        tree, tree_scens, tree_idxs, tree_vals = super().gen_tree(scens, start_tree)
        tree, tree_scens, tree_vals = self.init_vals(tree, tree_scens, tree_vals, scens)
        k = 0
        rel_dev = 1
        if do_plot:
            fig, ax = plt.subplots(1, 1)
        while rel_dev > tol and k < k_max:
            if k%1==0:
                loss = self.metric_loss(tree_vals, tree_idxs, scens)
                print('iter {}, loss: {}'.format(k, loss))

            if do_plot and k%1 == 0:
                ax.cla()
                replace_var(tree, tree_vals)
                plot_from_graph(tree, ax=ax, color=self.cm(2))
                ax.plot(scens, alpha=0.15, color=self.cm(5))
                ax.set_xlim(0, scens.shape[0] - 1)
                plt.title('loss: {:0.3}'.format(loss))
                plt.savefig(join(self.savepath, 'step_{:03d}'.format(k)))
                plt.pause(0.01)
            g = grad(partial(self.metric_loss, tree_idxs=tree_idxs, scens=scens))(tree_vals)
            tree_vals -= g*0.2
            k +=1

        replace_var(tree, tree_vals)
        return tree


@jit
def metric_loss(tree_vals, tree_idxs, scens):
    tot_dist = jnp.array(0)
    tree_scens = jnp.vstack([tree_vals[i] for i in tree_idxs.T]).T
    for scen in scens.T:
        dists = NeuralScenarioTree.compute_distances(tree_scens, scen)
        expdists = jnp.exp(80/(1 + dists))
        softmax = expdists / jnp.sum(expdists)
        tot_dist += jnp.sum(softmax * dists)
    return tot_dist


