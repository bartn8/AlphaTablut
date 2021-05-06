import ray
import numpy as np
from tablutconfig import TablutConfig
from tablut import Search, AshtonTablut, ActionStore, HeuristicFunction, NeuralHeuristicFunction, ptime


class Metrics:
    def __init__(self, nodes_explored = 0, max_depth = 0):
        self.nodes_explored = nodes_explored
        self.max_depth = max_depth

    def node_explored(self):
        self.nodes_explored += 1

    def update_max_depth(self, depth):
        if depth > self.max_depth:
            self.max_depth = depth

    def update_from_metric(self, metric):
        self.nodes_explored += metric.nodes_explored
        self.max_depth = max(self.max_depth, metric.max_depth)


class BranchResult:
    def __init__(self, metric, a, v, compute_time):
        self.metric = metric
        self.a = a
        self.v = v
        self.compute_time = compute_time

    def __str__(self):
        return "Action: {0}, Score: {1}, Nodes Explored: {2}, Max Depth: {3}, Time: {4}".format(self.a, self.v, self.metric.nodes_explored, self.metric.max_depth, self.compute_time)


@ray.remote
class BrachActor:

    def __init__(self, config, heuristic="none"):
        self.heuristic = HeuristicFunction.builder(heuristic, config)
        self.metric = Metrics()
        self.start_time = 0
        self.cutoff_time = 0
        self.current_cutoff_depth = 0

    def heuristic_initialized(self):
        return self.heuristic.initialized()

    def evalutate(self, board, player, turn):
        state = AshtonTablut.parse_board(board.copy(), player, turn)
        print(state.display())
        return self.heuristic.evalutate(state, player)

    def max_value(self, state, player, alpha, beta, depth):
        v = -np.inf
        terminal = state.terminal_test()

        self.metric.node_explored()
        self.metric.update_max_depth(depth)

        if depth > self.current_cutoff_depth or terminal or (ptime()-self.start_time) > self.cutoff_time:
            if terminal:
                return state.utility(player)
            return self.heuristic.evalutate(state, player)

        actions = state.actions()

        for action in actions:
            next_state = state.result(action)
            v = max(self.min_value(next_state, player, alpha, beta, depth + 1), v)
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, state, player, alpha, beta, depth):
        v = np.inf
        terminal = state.terminal_test()

        self.metric.node_explored()
        self.metric.update_max_depth(depth)

        if depth > self.current_cutoff_depth or terminal or (ptime()-self.start_time) > self.cutoff_time:
            if terminal:
                return state.utility(player)
            return self.heuristic.evalutate(state, player)

        actions = state.actions()

        for action in actions:
            next_state = state.result(action)
            v = min(self.max_value(next_state, player, alpha, beta, depth + 1), v)
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def min_branch_worker(self, a, board, to_move, turn, player, alpha, beta, depth, current_cutoff_depth, start_time, cutoff_time):
        state = AshtonTablut.parse_board(board.copy(), to_move, turn)
        self.metric = Metrics()
        self.current_cutoff_depth = current_cutoff_depth
        self.start_time = start_time
        self.cutoff_time = cutoff_time
        v = np.inf
        terminal = state.terminal_test()
        local_start_time = ptime()

        if depth > self.current_cutoff_depth or terminal or (ptime()-self.start_time) > self.cutoff_time:
            if terminal:
                return BranchResult(self.metric, a, state.utility(player), ptime()-local_start_time)
            return BranchResult(self.metric, a, self.heuristic.evalutate(state, player), ptime()-local_start_time)
        
        actions = state.actions()

        for action in actions:
            next_state = state.result(action)
            v = min(self.max_value(next_state, player, alpha, beta, depth + 1), v)
            if v <= alpha:
                return BranchResult(self.metric, a, state, v)
            beta = min(beta, v)

        return BranchResult(self.metric, a, v, ptime()-local_start_time)

    def cython_min_branch_worker(self, a, board, to_move, turn, player, alpha, beta, depth, current_cutoff_depth, start_time, cutoff_time):
        state = AshtonTablut.parse_board(board.copy(), to_move, turn)
        search = Search(self.heuristic)

        self.current_cutoff_depth = current_cutoff_depth
        self.start_time = start_time
        self.cutoff_time = cutoff_time
        
        local_start_time = ptime()

        v = search.min_branch(state, player, alpha, beta, depth, current_cutoff_depth, start_time, cutoff_time)

        self.metric = Metrics(search.nodes_explored, search.max_depth)
        
        return BranchResult(self.metric, a, v, ptime()-local_start_time) 


class MultiThreadSearch:

    def __init__(self, config, heuristic="none"):
        if config.cores <= 0:
            raise Exception("At least one node.")

        self.nodes = []
        self.n_nodes = config.cores
        self.heuristic = HeuristicFunction.builder(heuristic, config)
        for i in range(self.n_nodes):
            actor = BrachActor.remote(config, heuristic)
            initialized = ray.get(actor.heuristic_initialized.remote())
            if not initialized:
                raise Exception("Init heuristic failed")
            self.nodes.append(actor)

    def iterative_deepening_search(self, state, initial_cutoff_depth=2, cutoff_time=58.0):
        player = state.to_move()
        best_score = -np.inf
        beta = np.inf
        store = ActionStore()
        metric = Metrics()
        start_time = ptime()
        actions = state.actions()
        current_cutoff_depth = initial_cutoff_depth
        workers = {}
        best_action = None
        best_next_state = None

        for action in actions:
            next_state = state.result(action)
            v = self.heuristic.evalutate(next_state, player)
            store.add(action, v)

        timeout_occurred = (ptime()-start_time) > cutoff_time

        while not timeout_occurred:
            next_store = ActionStore()

            i = 0
            while i < (store.size()) and not timeout_occurred :
                action = store.actions[i]
                next_state = state.result(action)

                worker_id = -1

                for k in range(self.n_nodes):
                    if k not in workers:
                        worker_id = k
                        break

                if worker_id >= 0:
                    job_id = self.nodes[worker_id].cython_min_branch_worker.remote(action, next_state.board(
                    ), next_state.to_move(), next_state.turn(), player, best_score, beta, 1, current_cutoff_depth, start_time, cutoff_time)
                    workers[worker_id] = job_id
                    #i-esimo elemento schedulato
                    i+=1

                ready_ids, remaining_ids = ray.wait(
                    list(workers.values()), timeout=0)

                if len(ready_ids) > 0:
                    for job_id in ready_ids:
                        result = ray.get(job_id)
                        rMetric, rAction, rScore = result.metric, result.a, result.v
                        metric.update_from_metric(rMetric)
                        #print(result)

                        timeout_occurred = (
                            ptime()-start_time) > cutoff_time

                        if timeout_occurred:
                            break

                        if rScore > -1.0:
                            next_store.add(rAction, rScore)

                    next_workers = {}
                    for worker_id in list(workers.keys()):
                        if workers[worker_id] in remaining_ids:
                            next_workers[worker_id] = workers[worker_id]
                    workers = next_workers

                if timeout_occurred:
                    break

            if next_store.size() > 0:
                store = next_store
                timeout_occurred = (ptime()-start_time) > cutoff_time
                if not timeout_occurred:
                    action = store.actions[0]
                    score = store.utils[0]
                    if score >= 1.0:
                        return state.result(action), action, score, metric.max_depth, metric.nodes_explored, (ptime()-start_time)

            current_cutoff_depth += 1
            timeout_occurred = (ptime()-start_time) > cutoff_time

        if store.size() > 0:
            best_action = store.actions[0]
            best_score = store.utils[0]
            best_next_state = state.result(best_action)

        return best_next_state, best_action, best_score, metric.max_depth, metric.nodes_explored, (ptime()-start_time)


def general_test():
    ray.init()

    # Network loading
    config = TablutConfig()

    # Test network loading
    heuristic_name = "oldschool"

    #heuristic_test = NeuralHeuristicFunction(config)
    #if heuristic_test.init_tflite():
    #    print("Netowrk loaded successfully")
    #    heuristic_name = "neural"
    #else:
    #    print("Netowrk loading error")

    search = MultiThreadSearch(config, heuristic_name)

    state = AshtonTablut.get_initial()

    best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.iterative_deepening_search(
        state=state, initial_cutoff_depth=2, cutoff_time=10.0)

    best_action = AshtonTablut.num_to_coords(best_action)
    print("Game move ({0}): {1} -> {2}, Search time: {3}, Max Depth: {4}, Nodes explored: {5}, Score: {6}".format(
        state.to_move(),
        (best_action[0], best_action[1]),
        (best_action[2], best_action[3]),
        search_time,
        max_depth,
        nodes_explored,
        best_score))


def test_h():
    ray.init()

    # Network loading
    config = TablutConfig()

    # Test network loading
    heuristic_name = "oldschool"

    heuristic_test = NeuralHeuristicFunction(config)
    if heuristic_test.init_tflite():
        print("Netowrk loaded successfully")
        heuristic_name = "neural"
    else:
        print("Netowrk loading error")

    actor = BrachActor.remote(config, heuristic_name)
    print(ray.get(actor.heuristic_initialized.remote()))

    state = AshtonTablut.get_initial()
    print(ray.get(actor.evalutate.remote(
        state.board(), state.to_move(), state.turn())))


if __name__ == '__main__':
    general_test()
