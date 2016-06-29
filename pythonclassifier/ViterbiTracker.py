__author__ = 'Daniele'

import numpy as np
import networkx as nx
import scipy.spatial.distance as sd
import scipy.integrate as scint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ViterbiTracker():
    def __init__(self, detections):
        self.detections = detections

    def start(self, outfile, num_gap_frames):
        #f = open(outfile, mode='w', buffering=0)
        f = open(outfile, mode='w')

        self.state_space_diagram = self.construct_state_space_diagram(self.detections, num_gap_frames)
        self.initialize_weight_edges()
        F = [[]]

        g_F = 0
        last_g_F = -1
        while g_F > last_g_F:
            last_g_F = g_F

            diagram = self.state_space_diagram
            track, g_F, _len = viterbi(diagram, g_F)
            if _len == 1:
                self.update_weight_edges(F + track)
                continue
            print(track)

            if track in F:
                print("Algorithm ends.")
                g_F = -1
                continue
            else:
                F.append(track)

                track_vals = []
                for state in track:
                    if state[0].isdigit():
                        spltst = state.split('_')
                        center = self.state_space_diagram.node[state]['center']
                        track_vals.append((int(spltst[0]) + 1, int(center[0]), int(center[1])))

                f.write(str(track_vals)+'\n')

            self.update_weight_edges(F)

        f.close()

    def construct_state_space_diagram(self, detections, t_max=1):
        if t_max < 1:
            print("ERROR: t_max must be larger or equal one.")
            return -1

        g = nx.DiGraph()

        # start with the state A
        g.add_node('A')

        previous_t_max_node_lists = [[] for i in range(t_max)]
        previous_t_max_node_lists[0] = ['A']

        for t, detection_list in enumerate(detections):
            current_node_list = list()
            for n, detection in enumerate(detection_list):
                index = str(t) + '_' + str(n)
                current_node_list.append(index)
                g.add_node(index, center=list(detection))

                # add an edge for each node in the t_max previous detection lists and
                # automatically add a new node to the graph
                for previous_node_list in previous_t_max_node_lists:
                    for prev_node in previous_node_list:
                        g.add_edge(prev_node, index)

                        # if not the first frame, add 'dead state' connections
                        dead_state = 'X' + str(t)
                        if t != 0:
                            g.add_edge(prev_node, dead_state)

                # add edge with the previous 'born later' state
                # skip if first frame
                if t != 0:
                    born_later = 'Y' + str(t-1)
                    g.add_edge(born_later, index)

            # create singular born later and dead state connections
            if t > 0:
                g.add_edge('Y' + str(t-1), 'Y' + str(t))
            else:
                g.add_edge('A', 'Y' + str(t))
            if t > 1:
                g.add_edge('X' + str(t-1), 'X' + str(t))

            # update previous node lists
            previous_t_max_node_lists = previous_t_max_node_lists[1:] + [current_node_list]

        # end with connecting the state B
        last_node_list = previous_t_max_node_lists[-1]
        for final_node in last_node_list + ['Y'+str(t)] + ['X'+str(t)]:
            g.add_edge(final_node, 'B')

        return g

    def calculate_g(self, F):
        g = self.state_space_diagram
        _sum = 0
        for track in F:
            for i in xrange(0, len(track) - 1):
                _sum += g.edge[track[i]][track[i+1]]

        return _sum

    def initialize_weight_edges(self):
        g = self.state_space_diagram
        #print(len(g.edge))

        for edge in g.edge:
            if edge == 'A':
                list_nodes = g.successors('A')
                for succ_node in list_nodes:
                    if succ_node[0] is 'Y':
                        g.edge['A'][succ_node] = 0.01
                    else:
                        g.edge['A'][succ_node] = np.log(1 + _prob_C(1)) - np.log(1 + _prob_C(0))

            # edges from born later states
            elif edge[0] is 'Y':
                list_nodes = g.successors(edge)
                for succ_node in list_nodes:
                    if (succ_node[0] is not 'B') and (succ_node[0] is not 'Y'):
                        splst = succ_node.split('_')
                        t = int(splst[0])
                        i = int(splst[1])

                        k = int(edge.split('Y')[1])

                        x_t = g.node[succ_node]['center']

                        C_F = 0

                        # add new cell random appearance probability
                        g.edge[edge][succ_node] = np.log(1 + _prob_I(1, x_t)) +\
                                                  np.log(1 + _prob_C(C_F + 1)) - np.log(1 + _prob_C(C_F))
                    else:
                        g.edge[edge][succ_node] = 0.1

            elif edge[0] is 'X':
                list_nodes = g.successors(edge)
                for succ_node in list_nodes:
                    g.edge[edge][succ_node] = 0.

            # edges from detections
            else:
                list_nodes = g.successors(edge)
                for succ_node in list_nodes:
                    splst = edge.split('_')
                    t = int(splst[0])
                    i = int(splst[1])

                    if (succ_node[0] is not 'B') and (succ_node[0] is not 'X'):
                        splst = succ_node.split('_')
                        t_p = int(splst[0])
                        j = int(splst[1])

                        tau = t_p - t

                        x_t = g.node[edge]['center']
                        x_t_p = g.node[succ_node]['center']

                        C_F = _C_(t_p, j)

                        pre = 0.9 # TODO
                        f_u = 1.0/(512*512)
                        M_d = _prob_M(1, x_t_p, x_t, pre, f_u)
                        # REMOVE THE EDGE IF THE PROBABILITY OF MIGRATION IS ZERO
                        if M_d == 0:
                            g.remove_edge(edge, succ_node)
                        else:
                            g.edge[edge][succ_node] = np.log(1 + (1.0 / tau) * M_d) +\
                                                      np.log(1 + _prob_C(C_F + 1)) - np.log(1 + _prob_C(C_F))
                    else:
                        if succ_node[0] == 'X':
                            x_t = g.node[edge]['center']
                            t_p = int(succ_node.split('X')[1])

                            g.edge[edge][succ_node] = np.log(1 + _prob_I(1, x_t))
                        elif succ_node[0] == 'B':
                            g.edge[edge][succ_node] = 0.

    def update_weight_edges(self, F):
        g = self.state_space_diagram

        for edge in g.edge:
            # edges starting from A
            if edge is 'A':
                list_nodes = g.successors('A')
                for succ_node in list_nodes:
                    if succ_node[0].isdigit():
                        splst = succ_node.split('_')
                        t = int(splst[0])
                        i = int(splst[1])

                        C_F = _C_(t, i)
                        g.edge['A'][succ_node] = np.log(1 + _prob_C(C_F + 1)) - np.log(1 + _prob_C(C_F))

            # edges from born later states
            elif edge[0] is 'Y':
                list_nodes = g.successors(edge)
                for succ_node in list_nodes:
                    if (succ_node[0] is not 'B') and (succ_node[0] is not 'Y'):
                        splst = succ_node.split('_')
                        t = int(splst[0])
                        i = int(splst[1])

                        k = int(edge.split('Y')[1])

                        x_t = g.node[succ_node]['center']

                        C_F = _C_(t, i)

                        # add new cell random appearance probability TODO
                        g.edge[edge][succ_node] = np.log(1 + _prob_I(1, x_t)) - np.log(1 + _prob_I(_I_y(k,t,i,F), x_t)) +\
                                                  np.log(1 + _prob_C(C_F + 1)) - np.log(1 + _prob_C(C_F))


            elif edge[0] is 'X':
                # do nothing
                None

            # edges from detections
            else:
                list_nodes = g.successors(edge)
                for succ_node in list_nodes:
                    splst = edge.split('_')
                    t = int(splst[0])
                    i = int(splst[1])

                    if (succ_node[0] is not 'B') and (succ_node[0] is not 'X'):
                        splst = succ_node.split('_')
                        t_p = int(splst[0])
                        j = int(splst[1])

                        tau = t_p - t

                        x_t = g.node[edge]['center']
                        x_t_p = g.node[succ_node]['center']

                        C_F = _C_(t_p, j)

                        pre = 0.9 # TODO
                        f_u = 1.0/(512*512)
                        g.edge[edge][succ_node] = np.log(1 + (1.0 / tau) * _prob_M(1, x_t_p, x_t, pre, f_u)) -\
                                                  np.log(1 + _prob_M(_M_(t, i, t_p, j, F), x_t_p, x_t, pre, f_u)) +\
                                                  np.log(1 + _prob_C(C_F + 1)) - np.log(1 + _prob_C(C_F))
                    else:
                        if succ_node[0] == 'X':
                            x_t = g.node[edge]['center']
                            t_p = int(succ_node.split('X')[1])

                            g.edge[edge][succ_node] = np.log(1 + _prob_I(1, x_t)) - np.log(1 + _prob_I(_I_(t, i, t_p, F), x_t))

        return 0

f_cache = np.ones((16, 16))*-1

_C_counts = dict()

def viterbi(diagram, g_F):
    # for every t in [1:T+1]
    g = diagram

    next_states = ['A']
    g_max_prec = g_F
    _len = 0
    curr_state = 'A'
    succ_states = g.succ[curr_state]
    while len(succ_states) > 0:
        curr_max = 0
        next_state_idx = -1
        for idx_state, succ in enumerate(succ_states):
            delta_g = g.edge[curr_state][succ]
            if g_max_prec + delta_g > curr_max:
                curr_max = g_max_prec + delta_g
                next_state_idx = idx_state

        next_state = ''
        for idx_state, succ in enumerate(succ_states):
            if idx_state == next_state_idx:
                next_state = succ

        if next_state[0].isdigit():
            _len += 1
            if _C_counts.has_key(next_state):
                val = int(_C_counts.get(next_state))
                _C_counts[next_state] = val + 1
            else:
                _C_counts[next_state] = 1

            if (next_states[-1])[0].isdigit():
                _M_counts[(next_states[-1] + ',' + next_state)] = True
            elif (next_states[-1])[0] == 'Y':
                _I_y_counts[(next_states[-1] + ',' + next_state)] = True
        elif next_state[0] == 'X':
            if (next_states[-1])[0].isdigit():
                _I_counts[(next_states[-1] + ',' + next_state)] = True

        next_states.append(next_state)
        succ_states = g.succ[next_state]
        g_max_prec = curr_max
        curr_state = next_state

    return next_states, g_max_prec, _len

prob_C_cache = dict()

def _prob_C(x):
    if prob_C_cache.has_key(x):
        return prob_C_cache.get(x)
    else:
        #val = stat.poisson(1.3).pmf(x)
        if x == 0:
            val = 0.2
        elif x == 1:
            val = 0.5
        elif x > 1:
            val = 0.05
        prob_C_cache[x] = val
        return val


def _C_(t, i):
    det = str(t) + '_' + str(i)
    if _C_counts.has_key(det):
        return int(_C_counts.get(det))
    else:
        return 0
    # _sum = 0
    # for track in F:
    #     try:
    #         el = track.index(str(t) + '_' + str(i))
    #     except ValueError:
    #         continue
    #
    #     _sum += 1
    # return _sum

prob_I_cache = dict()


def _prob_I(v, x_t):
    if v == 1:
        if prob_I_cache.has_key(tuple(x_t)):
            return prob_I_cache.get(tuple(x_t))
        else:
            sigma = 4 # todo change this

            delta = 2
            span = sigma*4
            x_range = range(1, span, delta)
            k = len(x_range)
            matx = np.zeros((k, k))
            for i in x_range:
                for j in x_range:
                    i_f = x_t[0] - (span/2) + i
                    j_f = x_t[1] - (span/2) + j

                    if (np.array([i_f, j_f]) < 0).any() or (np.array([i_f, j_f]) >= 512).any():
                        matx[i/delta, j/delta] = 0
                    else:
                        matx[i/delta, j/delta] = f([i_f, j_f], x_t, sigma)

            first = scint.simps(matx, x_range, axis=0)
            _sum = scint.simps(first, x_range)
            val = 1-_sum
            prob_I_cache[tuple(x_t)] = val
            return val
    else:
        return 0


def f(x_t_p, x_t, sigma):
    diff = np.abs(np.subtract(x_t_p, x_t))
    if (diff >= sigma * 2).any():
        return 0.
    if f_cache[diff[0], diff[1]] != -1:
        return f_cache[diff[0], diff[1]]
    else:
        distance = (1/(2*np.pi*(sigma ** 2))) * np.exp((-sd.sqeuclidean(x_t_p, x_t))/(2 * (sigma ** 2)))
        f_cache[diff[0], diff[1]] = distance
        return distance


def _prob_M(m, x_t_p, x_t, pre, f_u):
    if m == 1:
        return (pre*f(x_t_p, x_t, 4))/(pre*f(x_t_p, x_t, 4) + (1-pre)*f_u)
    elif m == 0:
        return 0
    else:
        print('ERROR')

_M_counts = dict()


def _M_(t, i, t_p, j, F):
    d1 = str(t) + '_' + str(i)
    d2 = str(t_p) + '_' + str(j)
    d = d1 + ',' + d2
    return d in _M_counts

_I_counts = dict()


def _I_(t, i, t_p, F):
    d1 = str(t) + '_' + str(i)
    dx = 'X' + str(t_p)
    d = d1 + ',' + dx
    return d in _I_counts

_I_y_counts = dict()


def _I_y(k, t, i, F):
    dy = 'Y' + str(k)
    d2 = str(t) + '_' + str(i)
    d = dy + ',' + d2
    return d in _I_y_counts

# main used for presentation purposes only
if __name__ == '__main__':
    string_res = "tracks_pres.txt"

    vt = ViterbiTracker(np.load("detections_50.npy"))
    vt.start(string_res, 2)
    tracks_file = open(string_res, "r")
    ind = 0
    tracks = list()
    for line in tracks_file:
        track_arr = np.array(eval(line))
        if len(track_arr > 1):
            tracks.append(track_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for track in tracks:
        ax.plot(track[:,1], track[:,2], track[:,0])
    plt.show()