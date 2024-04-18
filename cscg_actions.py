from __future__ import print_function
from builtins import range
import numpy as np
# import numba as nb
from tqdm import trange
import sys
import networkx as nx  # Import the networkx library
# from numba import jit
# import numpy as np
import matplotlib.pyplot as plt
import pdb
from IPython.display import display, Image
import pickle

# from __future__ import print_function
# from builtins import range
# import numpy as np
# import numba as nb
# from tqdm import trange
# import sys
# import networkx as nx  # Import the networkx library
# from numba import jit



def validate_seq(x, a, n_clones=None):
    """Validate an input sequence of observations x and actions a"""
    assert len(x) == len(a) > 0
    assert len(x.shape) == len(a.shape) == 1, "Flatten your array first"
    assert x.dtype == a.dtype == np.int64
    assert 0 <= x.min(), "Number of emissions inconsistent with training sequence"
    if n_clones is not None:
        assert len(n_clones.shape) == 1, "Flatten your array first"
        assert n_clones.dtype == np.int64
        assert all(
            [c > 0 for c in n_clones]
        ), "You can't provide zero clones for any emission"
        n_emissions = n_clones.shape[0]
        assert (
            x.max() < n_emissions
        ), "Number of emissions inconsistent with training sequence"


def datagen_structured_obs_room(
    room,
    start_r=None,
    start_c=None,
    no_left=[],
    no_right=[],
    no_up=[],
    no_down=[],
    length=10000,
    seed=42,
):
    """room is a 2d numpy array. inaccessible locations are marked by -1.
    start_r, start_c: starting locations

    In addition, there are invisible obstructions in the room
    which disallows certain actions from certain states.

    no_left:
    no_right:
    no_up:
    no_down:

    Each of the above are list of states from which the corresponding action is not allowed.

    """
    np.random.seed(seed)
    H, W = room.shape
    if start_r is None or start_c is None:
        start_r, start_c = np.random.randint(H), np.random.randint(W)

    actions = np.zeros(length, int)
    x = np.zeros(length, int)  # observations
    rc = np.zeros((length, 2), int)  # actual r&c

    r, c = start_r, start_c
    x[0] = room[r, c]
    rc[0] = r, c

    count = 0
    while count < length - 1:

        act_list = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
        if (r, c) in no_left:
            act_list.remove(0)
        if (r, c) in no_right:
            act_list.remove(1)
        if (r, c) in no_up:
            act_list.remove(2)
        if (r, c) in no_down:
            act_list.remove(3)

        a = np.random.choice(act_list)

        # Check for actions taking out of the matrix boundary.
        prev_r = r
        prev_c = c
        if a == 0 and 0 < c:
            c -= 1
        elif a == 1 and c < W - 1:
            c += 1
        elif a == 2 and 0 < r:
            r -= 1
        elif a == 3 and r < H - 1:
            r += 1

        # Check whether action is taking to inaccessible states.
        temp_x = room[r, c]
        if temp_x == -1:
            r = prev_r
            c = prev_c
            pass

        actions[count] = a
        x[count + 1] = room[r, c]
        rc[count + 1] = r, c
        count += 1

    return actions, x, rc

# @nb.njit
class TableContainer:
    def __init__(self):
        self.groups_of_tables = {}
        self.table_totals = {}  # Keep track of totals for each table separately
        self.total_observations = 0  # Keep track of total observations across all tables

    def add_clone(self, group_id, table_id):
        """Add exactly one clone to a specified table, creating the table or group if necessary."""
        # Automatically create the group and table if they don't exist
        if group_id not in self.groups_of_tables:
            self.groups_of_tables[group_id] = {}
        if table_id not in self.groups_of_tables[group_id]:
            self.groups_of_tables[group_id][table_id] = 0  # Initialize clones count for the table

        # Add one clone to the table count and update total observations
        self.groups_of_tables[group_id][table_id] += 1
        self.table_totals[(group_id, table_id)] = self.groups_of_tables[group_id][table_id]  # Update table total
        self.total_observations += 1

    def get_total_observations(self):
        """Return the total number of observations."""
        return self.total_observations

    def get_group_total(self, group_id):
        """Return the total number of clones in all tables within a specific group."""
        return sum(self.groups_of_tables.get(group_id, {}).values())

    def get_table_total(self, group_id, table_id):
        """Return the total number of clones for a specific table."""
        return self.groups_of_tables.get(group_id, {}).get(table_id, 0)

    def count_tables_in_group(self, group_id):
        """Returns the number of tables within the specified group."""
        if group_id in self.groups_of_tables:
            return len(self.groups_of_tables[group_id])
        else:
            # print(f"Group {group_id} does not exist.")
            return 0

def CRP(container, curr_observation, alpha=1.0):
    """
    Simulates the Chinese Restaurant Process.

    Parameters:
    - history: int, the total number of customers to simulate.
    - alpha: float, the concentration parameter.

    Returns:
    - A list where the i-th element represents the table number of the i-th customer.
    """

    n = container.get_group_total(curr_observation)


    if curr_observation not in container.groups_of_tables:
      container.add_clone(curr_observation,0)
      table_choice = 0
      assignments = 0
      probs = 1
    else:
      probs = [clone_count / (n + alpha) for table_id, clone_count in container.groups_of_tables[curr_observation].items()] + [alpha / (n + alpha)] # This is the prior


    # Add an update rule (ref Nora's paper 1st equation)

    # Choose a table based on the probabilities
      table_choice = np.random.choice(len(probs), p=probs)


    # update clone --> existing or new, same
      container.add_clone(curr_observation,table_choice)
      assignments = table_choice

    return assignments, probs



class CHMM_LCM(object):
    def __init__(self, x, a, container, n_clones=1, pseudocount=0.0, alpha=1.0, dtype=np.float32, seed=42, filename='best_model.pkl'):
        """Construct a CHMM objct. n_clones is an array where n_clones[i] is the
        number of clones assigned to observation i. x and a are the observation sequences
        and action sequences, respectively."""
        np.random.seed(seed)
        n_states = len(np.unique(x))# self.n_clones.sum() # this should be changed too; for now, just define them as observations
        n_actions = a.max() + 1
        self.n_clones = n_states #n_clones
        # validate_seq(x, a, self.n_clones)
        assert pseudocount >= 0.0, "The pseudocount should be positive"
        # print("Average number of clones:", n_clones.mean())
        self.pseudocount = pseudocount
        self.dtype = dtype

        # self.C = np.random.rand(n_actions, n_states, n_states).astype(dtype) # this should be changed; actually, n_actions, n_states should be modified too
        self.C = np.random.rand(n_actions, n_states, n_states).astype(dtype) # this should be changed; actually, n_actions, n_states should be modified too

        self.Pi_x = np.ones(n_states) / n_states
        self.Pi_a = np.ones(n_actions) / n_actions
        self.update_T()
        self.container = container
        self.alpha = alpha
        self.best_loss = float('inf')
        self.best_model_filename = filename
        # self.progression = []
        # self.container = TableContainer() # initialize CRP
        # self.all_table_counts = {}
        # for i in range(1, len(n_clones) + 1):
        #     self.all_table_counts[i] = {1: 1}  # Each table initialized with {1: 1}


    def save_best_model(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss  # Update the best loss
            with open(self.best_model_filename, 'wb') as file:
                pickle.dump(self, file)
        # Save the model only if the current loss is the best one
        # if current_loss < self.best_loss:
            # self.best_loss = current_loss  # Update the best loss
            # with open(self.best_model_filename, 'wb') as f:
                # pickle.dump({'model': self, 'loss': self.best_loss}, f)
            # print(f"New best model saved with loss {current_loss}")
        # else:
            # print(f"Model not saved, current loss {current_loss} is not lower than best loss {self.best_loss}")

    def load_model(self):
        # Load the best model from a file
        with open(self.best_model_filename, 'rb') as f:
            loaded_model = pickle.load(f)
            # self.__dict__.update(data['model'].__dict__)
            # self.best_loss = data['loss']
        
        print(f"Model loaded with loss {self.best_loss}")
        return loaded_model
    
# with open('model_filename.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
    
    
    
    def update_T(self):
        """Update the transition matrix given the accumulated counts matrix."""
        self.T = self.C + self.pseudocount
        norm = self.T.sum(2, keepdims=True)
        norm[norm == 0] = 1
        self.T /= norm


    # def update_T(self):
    #     self.T = self.C + self.pseudocount
    #     norm = self.T.sum(2, keepdims=True)  # old model (conditional on actions)
    #     norm[norm == 0] = 1
    #     self.T /= norm
    #     norm = self.T.sum((0, 2), keepdims=True)  # new model (generates actions too)
    #     norm[norm == 0] = 1
    #     self.T /= norm

    def update_E(self, CE):
        """Update the emission matrix."""
        E = CE + self.pseudocount
        norm = E.sum(1, keepdims=True)
        norm[norm == 0] = 1
        E /= norm
        return E

    def bps(self, x, a):
        """Compute the log likelihood (log base 2) of a sequence of observations and actions."""
        validate_seq(x, a, self.n_clones)
        log2_lik = forward(self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a)[0]
        return -log2_lik

    def bpsE(self, E, x, a):
        """Compute the log likelihood using an alternate emission matrix."""
        validate_seq(x, a, self.n_clones)
        log2_lik = forwardE(
            self.T.transpose(0, 2, 1), E, self.Pi_x, self.n_clones, x, a
        )
        return -log2_lik

    def bpsV(self, x, a):
        validate_seq(x, a, self.n_clones)
        log2_lik = forward_mp(
            self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a
        )[0]
        return -log2_lik

    def decode(self, x, a):
        """Compute the MAP assignment of latent variables using max-product message passing."""
        # print("decode C: {}".format(np.shape(self.C)))
        # print("decode T: {}".format(np.shape(self.T)))
        # print("decode Pi: {}".format(np.shape(self.Pi_x)))
        # print("decode n_clones: {}".format(self.n_clones))
        log2_lik, mess_fwd = forward_mp(
            self.T.transpose(0, 2, 1),
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states, ginis = backtrace(self.T, self.n_clones, x, a, mess_fwd)
        # print(mess_fwd)
        return -log2_lik, states, ginis

    def decodeE(self, E, x, a):
        """Compute the MAP assignment of latent variables using max-product message passing
        with an alternative emission matrix."""
        log2_lik, mess_fwd = forwardE_mp(
            self.T.transpose(0, 2, 1),
            E,
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states = backtraceE(self.T, E, self.n_clones, x, a, mess_fwd)
        return -log2_lik, states

    def learn_em_T(self, x, a, n_iter=100, term_early=True):
        """Run EM training, keeping E deterministic and fixed, learning T"""
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        # pdb.set_trace()
        for it in pbar:
            # E
            log2_lik, mess_fwd, n_clones, T, mess_loc, state_loc, unique_obs = forward(
                self.T.transpose(0, 2, 1),
                self.Pi_x, # this seems to be initialized just at once (no further updates)
                self.n_clones,
                x,
                a,
                self.container,
                self.alpha,
                store_messages=True,
            )

            # update self.n_clones
            # update T
            # print("n_clones after forward: {}".format(n_clones)) ### PROBLEM HERE: ALWAYS RETURNS [1,1,1,1]
            self.n_clones = n_clones

            self.T = T
            mess_bwd = backward(self.T, self.n_clones, x, a, mess_loc, state_loc, unique_obs)
            self.C = updateC(self.C, self.T, self.n_clones, mess_fwd, mess_bwd, x, a, mess_loc, state_loc, unique_obs)
            # print("C shape after updateC: {}".format(np.shape(self.C))) # this is where C seems to be initialized -> updateC didn't resulted in change sef.C?
            # M
            self.update_T()
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            # loss = 
            if convergence[-1] <= self.best_loss: 
                self.save_best_model(convergence[-1])
                pbar.set_postfix_str(f"New best model at epoch {it} saved with loss {self.best_loss:.4f}")
            pbar.update(1)
        pbar.close()
                # 
            # progression = convergence
            # if log2_lik.mean() <= log2_lik_old:
            #     if term_early:
            #         break
            # log2_lik_old = log2_lik.mean()
            # print(log2_lik_old)
        # JY added for plotting learning curve
        # plt.plot(convergence, label='Training Loss')
        # plt.title('Learning Curve')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.show()
        self.progression = convergence
        return convergence

    def learn_viterbi_T(self, x, a, n_iter=100):
        """Run Viterbi training, keeping E deterministic and fixed, learning T"""
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        print("viterbi C: {}".format(np.shape(self.C)))
        print("viterbi T: {}".format(np.shape(self.T)))
        for it in pbar:
            # E
            log2_lik, mess_fwd = forward_mp(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            states,_ = backtrace(self.T, self.n_clones, x, a, mess_fwd)
            self.C[:] = 0
            for t in range(1, len(x)):
                aij, i, j = (
                    a[t - 1],
                    states[t - 1],
                    states[t],
                )  # at time t-1 -> t we go from observation i to observation j
                self.C[aij, i, j] += 1.0
            # M
            self.update_T()

            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        return convergence

    def learn_em_E(self, x, a, n_iter=100, pseudocount_extra=1e-20):
        """Run Viterbi training, keeping T fixed, learning E"""
        sys.stdout.flush()
        n_emissions, n_states = len(self.n_clones), self.n_clones.sum()
        CE = np.ones((n_states, n_emissions), self.dtype)
        E = self.update_E(CE + pseudocount_extra)
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for it in pbar:
            # E
            log2_lik, mess_fwd = forwardE(
                self.T.transpose(0, 2, 1),
                E,
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backwardE(self.T, E, self.n_clones, x, a)
            updateCE(CE, E, self.n_clones, mess_fwd, mess_bwd, x, a)
            # M
            E = self.update_E(CE + pseudocount_extra)
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        return convergence, E

    def sample(self, length):
        """Sample from the CHMM."""
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)
        sample_x = np.zeros(length, dtype=np.int64)
        sample_a = np.random.choice(len(self.Pi_a), size=length, p=self.Pi_a)

        # Sample
        p_h = self.Pi_x
        for t in range(length):
            h = np.random.choice(len(p_h), p=p_h)
            sample_x[t] = np.digitize(h, state_loc) - 1
            p_h = self.T[sample_a[t], h]
        return sample_x, sample_a

    def sample_sym(self, sym, length):
        """Sample from the CHMM conditioning on an inital observation."""
        # Prepare structures
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)

        seq = [sym]

        alpha = np.ones(self.n_clones[sym])
        alpha /= alpha.sum()

        for _ in range(length):
            obs_tm1 = seq[-1]
            T_weighted = self.T.sum(0)

            long_alpha = np.dot(
                alpha, T_weighted[state_loc[obs_tm1] : state_loc[obs_tm1 + 1], :]
            )
            long_alpha /= long_alpha.sum()
            idx = np.random.choice(np.arange(self.n_clones.sum()), p=long_alpha)

            sym = np.digitize(idx, state_loc) - 1
            seq.append(sym)

            temp_alpha = long_alpha[state_loc[sym] : state_loc[sym + 1]]
            temp_alpha /= temp_alpha.sum()
            alpha = temp_alpha

        return seq

    def bridge(self, state1, state2, max_steps=100):
        Pi_x = np.zeros(self.n_clones.sum(), dtype=self.dtype)
        Pi_x[state1] = 1
        log2_lik, mess_fwd = forward_mp_all(
            self.T.transpose(0, 2, 1), Pi_x, self.Pi_a, self.n_clones, state2, max_steps
        )
        s_a = backtrace_all(self.T, self.Pi_a, self.n_clones, mess_fwd, state2)
        return s_a


def updateCE(CE, E, n_clones, mess_fwd, mess_bwd, x, a):
    timesteps = len(x)
    gamma = mess_fwd * mess_bwd
    norm = gamma.sum(1, keepdims=True)
    norm[norm == 0] = 1
    gamma /= norm
    CE[:] = 0
    for t in range(timesteps):
        CE[:, x[t]] += gamma[t]


def forwardE(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """Log-probability of a sequence, and optionally, messages"""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = T_tr[aij].dot(message)
        message *= E[:, j]
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def backwardE(T, E, n_clones, x, a):
    """Compute backward messages."""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T.dtype.type

    # backward pass
    t = x.shape[0] - 1
    message = np.ones(E.shape[0], dtype)
    message /= message.sum()
    mess_bwd = np.empty((len(x), E.shape[0]), dtype=dtype)
    mess_bwd[t] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, j = (
            a[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        message = T[aij].dot(message * E[:, j])
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        mess_bwd[t] = message
    return mess_bwd


# @nb.njit
def updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a, mess_loc, state_loc, unique_obs):
    # state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    # mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    timesteps = len(x)
    # print("mess_loc: {}".format(mess_loc))

    # print("PRE update c: {}".format(np.shape(C)))
    # C[:] = 0
    C = np.zeros(np.shape(T))
    # print("len mess fwd: {}, len mess backward: {}".format(np.shape(mess_fwd), np.shape(mess_bwd)))
    # print("POST update c: {}".format(np.shape(C)))
    for t in range(1, timesteps):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (tm1_start, tm1_stop), (t_start, t_stop) = (
            mess_loc[t - 1 : t + 1],
            mess_loc[t : t + 2],
        )
        old_ind = unique_obs.index(i)
        ind = unique_obs.index(j) # observation index
        # print("tm1_start: {}, tm1_stop: {}".format(tm1_start, tm1_stop))
        # print("t_start: {}, t_stop: {}".format(t_start, t_stop))

        i_start = state_loc[old_ind]
        i_stop = state_loc[old_ind] + (tm1_stop - tm1_start)
        j_start = state_loc[ind]
        j_stop = state_loc[ind] + (t_stop - t_start)

        # print("T shape: {}".format(np.shape(T)))
        # print("state_loc: {}".format(state_loc))
        # print("i_start: {}, i_stop: {}".format(i_start, i_stop))
        # print("j_start: {}, j_stop: {}".format(j_start, j_stop))
        # print("len mess bwd: {}".format(np.shape(mess_bwd)))
        # print(np.shape(mess_fwd[tm1_start:tm1_stop].reshape(-1, 1)), np.shape(T[aij, i_start:i_stop, j_start:j_stop]), np.shape(mess_bwd[t_start:t_stop].reshape(1, -1)) )
        q = (
            mess_fwd[tm1_start:tm1_stop].reshape(-1, 1)
            * T[aij, i_start:i_stop, j_start:j_stop]
            * mess_bwd[t_start:t_stop].reshape(1, -1)
        )
        # print(q)
        q /= q.sum()
        C[aij, i_start:i_stop, j_start:j_stop] += q
    return C
    # print("C: {}".format(C))


def post_clone_T(matrix, insert_row, insert_col, new_element=None, placeholder=None):
    # If new_element is not specified, generate a random number using np.random.rand()
    if new_element is None:
        new_element = np.random.rand()
    # Determine the size of the original matrix
    num_slices, original_rows, original_cols = matrix.shape

    # Create a new matrix with additional row and column for each slice
    new_matrix_shape = (num_slices, original_rows + 1, original_cols + 1)
    if placeholder is None:
      new_matrix = np.random.rand(*new_matrix_shape)
    else:
      new_matrix = np.full(new_matrix_shape, placeholder, dtype=object)  # Use if a specific placeholder is required
    # if placeholder is None:
    #   # Create a new matrix with additional row and column
    #   new_matrix = [[np.random.rand() for _ in range(original_cols + 1)] for __ in range(original_rows + 1)]
    # else:
    #   new_matrix = [[placeholder for _ in range(original_cols + 1)] for __ in range(original_rows + 1)]

    for slice_index in range(num_slices):
        for i in range(original_rows + 1):
            for j in range(original_cols + 1):
                if i == insert_row and j == insert_col:
                    if new_element is not None:
                        # Insert the new element if specified
                        new_matrix[slice_index, i, j] = new_element
                        # print("Insert the new element if specified")
                elif i < insert_row and j < insert_col:
                    new_matrix[slice_index, i, j] = matrix[slice_index, i, j]
                elif i <= insert_row or j <= insert_col:
                    # For elements in the new row or column, keep the random value,
                    # or insert a placeholder if it's specified and we're not in the insertion cell
                    if placeholder is not None and (i == insert_row or j == insert_col):
                        new_matrix[slice_index, i, j] = placeholder
                        print("keep the random value")
                else:
                    # Adjust indices for copying from the original matrix
                    orig_i, orig_j = (i - 1 if i > insert_row else i), (j - 1 if j > insert_col else j)
                    new_matrix[slice_index, i, j] = matrix[slice_index, orig_i, orig_j]
                    # print("Adjust indices for copying from the original matrix")

    return new_matrix


def forward(T_tr, Pi, n_clones, x, a, container,alpha=1.0,store_messages=False):
  # JY changed for full bottom-up assignment of clones (i.e., no prior assumption about the number of observations)
    """Log-probability of a sequence, and optionally, messages"""

    # Assumption: no prior assumption about the observations, and assume no clones (clones will only be created after direct experience;
    # but interesting to assume transfer learning (i.e., clone creation in a similar observation will lead to creation of another similar observation?) )
    # T_tr = self.T

    # Pi: nclone * nobservations (e.g., 2 clones * 5 observations == 10 states)
    dtype = T_tr.dtype.type


    # Assume a blank state of observations:
    # create an empty list to store unique observations
    unique_obs = []
    # create an empty list to store number of clones

    # Don't initialize after 1st iteration
    # print(container.get_total_observations())
    if container.get_total_observations() == 0:
      # print('initialized')
      n_clones = np.array([], dtype=np.int64)
    mess_loc = np.array([0]) # np.array([], dtype=dtype)
    mess_fwd = np.array([0]) # np.array([], dtype=dtype)

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t] # this is the first observation

    # Loop through the observations and add unique elements to the list
    if j not in unique_obs:
      unique_obs.append(j)

    # This is where CRP (clone separation) happens
    # def CRP(n, table_counts, curr_observation, alpha=1.0):
    ind = unique_obs.index(j)
    prev_tables = container.count_tables_in_group(ind)
    # pdb.set_trace()
    # print(prev_tables)

    assignment, _ =  CRP(container, ind, alpha=alpha)
    post_tables = container.count_tables_in_group(ind)
    # print(post_tables)

    if prev_tables != post_tables: # a new clone has been created for this observation

      # n_clones[j-1] += 1
      # if ind >= len(n_clones): # n_clones[ind] does not exist
      n_clones = np.append(n_clones, 1)
      # print('splitted')
      # n_clones = np.concatenate((n_clones, np.array([0])))  # Append an initial array
      # n_clones[ind] += int(1)
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum() # redefine start # of the states (observations)
    # print(state_loc)
      # if store_messages:
      #   mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()

    # now update clone matrix if different clone has been created

    # JY: Update Pi (initial probability) according to the clone
    n_states = n_clones.sum()
    # print(n_states)
    Pi = np.ones(n_states) / n_states
    # print(Pi)

    # j_start, j_stop = state_loc[j : j + 2]
    j_start, j_stop = state_loc[ind : ind + 2]

    message = Pi[j_start:j_stop].copy().astype(dtype)
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        # mess_loc = np.hstack(
        #     (np.array([0], dtype=n_clones.dtype), n_clones[x])
        # ).cumsum()
        # mess_fwd = np.empty(mess_loc[-1], dtype=dtype)
        # t_start, t_stop = mess_loc[t : t + 2]
        # mess_fwd[t_start:t_stop] = message

        # All of the above should be changed
        mess_loc = np.append(mess_loc, n_clones[ind])
        mess_fwd = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message

    # else:
    #     mess_fwd = None

    # T_tr should be modified too

    for t in range(1, x.shape[0]):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j

        # 1. Loop through the observations and add unique elements to the list
        if j not in unique_obs:
          unique_obs.append(j)
        old_ind = ind # previous index
        ind = unique_obs.index(j) # observation index

        # This is where CRP (clone separation) happens
        prev_tables = container.count_tables_in_group(ind)
        # print(prev_tables)
        if old_ind != ind: # DON'T SPLIT CLONES WHEN TRANSITIONING TO ITSELF
          assignment, _ =  CRP(container, ind, alpha=alpha)
        post_tables = container.count_tables_in_group(ind)
        # print(post_tables)

        if prev_tables != post_tables: # a new clone has been created for this observation
          if ind >= len(n_clones): # n_clones[ind] does not exist
            n_clones = np.append(n_clones, 0)
          n_clones[ind] += 1
            # n_clones = np.concatenate((n_clones, np.array([0])))  # Append an initial array
          # n_clones[ind] += 1
          # Insert T_tr?
          # T_tr[aij,]

        state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum() # redefine start # of the states (observations)

        (i_start, i_stop), (j_start, j_stop) = (
            # state_loc[i : i + 2],
            # state_loc[j : j + 2],
            state_loc[old_ind : old_ind + 2],
            state_loc[ind : ind + 2],
        )

        if prev_tables != post_tables: # a new clone has been created for this observation
          if n_clones[ind] > 1: # don't append table when the room was just created
          # Transition matrix
          # matrix = T_tr[aij,:,:]
            T_tr = post_clone_T(T_tr, state_loc[ind], state_loc[ind])

            # Should normalize: given an action (dimension 0) and state t (dimension 2), the sum of the next states (t+1) should be 1
            # self.T = self.C + self.pseudocount
            norm = T_tr.sum(1, keepdims=True)
            norm[norm == 0] = 1
            T_tr /= norm




        # print("post_clone_T shape: {}".format(np.shape(T_tr)))
        # print("message shape: {}".format(np.shape(message)))
          # print(np.shape(T_tr))




        # Now that we have a[t-1], x[t-1], x[t] (and the latent states), update the transition matrix accordingly
        # np.random.rand(n_actions, 1, 1)
        #### PICK UP LATER: TRANSITION MATRIX!!! (HOW TO DYNAMICALLY UPDATE ACCORDING TO NEW TRANSITIONS)
        # print("old_ind: {}, new_ind: {}".format(old_ind, ind))
        # print("N clones: {}".format(n_clones))
        # print("T_tr = {}".format(np.shape(T_tr[aij, j_start:j_stop, i_start:i_stop])))
        # print("Message: {}".format(message))
        message = np.ascontiguousarray(T_tr[aij, j_start:j_stop, i_start:i_stop]).dot(
            message
        )
        # print("Message after dot: {}".format(message))
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            # mess_loc = np.append(mess_loc, n_clones[ind])
            mess_loc = np.append(mess_loc, len(message))
            # print('mess_loc: {}'.format(mess_loc))
            t_start, t_stop = mess_loc[t : t + 2]
            # print('message: {}'.format(message))
            mess_fwd = np.append(mess_fwd, message)
            # mess_fwd[t_start:t_stop] = message
            # print(mess_loc, len(mess_fwd))
    # print(mess_fwd)
    # print(np.shape(mess_fwd))
    mess_loc = np.cumsum(mess_loc)
    return log2_lik, mess_fwd, n_clones, T_tr, mess_loc, state_loc, unique_obs #, state_loc, Pi # JY added state_loc and Pi for updated clones

def backward(T, n_clones, x, a, mess_loc, state_loc, unique_obs):
    """Compute backward messages."""
    # pdb.set_trace()
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T.dtype.type
    # old_ind = unique_obs.index(i)
    # ind = unique_obs.index(j) # observation index

    # backward pass
    t = x.shape[0] - 1
    i = x[t]

    ind = unique_obs.index(i)
    message = np.ones(n_clones[ind], dtype) / n_clones[ind]
    message /= message.sum()
    # mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    mess_bwd = np.empty(mess_loc[-1], dtype)
    t_start, t_stop = mess_loc[t : t + 2]
    mess_bwd[t_start:t_stop] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        if t == x.shape[0]:

          # (tm1_start, tm1_stop), (t_start, t_stop) = (
          #     mess_loc[t:t+2],
          #     mess_loc[t+1],
          # )
          (tm1_start, tm1_stop) = mess_loc[t:t+2]
          t_start = mess_loc[t+1]
          t_stop = mess_loc[t+1] + n_clones[ind]
        else:
           (tm1_start, tm1_stop), (t_start, t_stop) = (
              mess_loc[t : t+2 ],
              mess_loc[t+1 : t + 3],
          )



        # t_start, t_stop = mess_loc[t : t + 2]

        old_ind = unique_obs.index(x[t])
        ind = unique_obs.index(x[t+1])

        i_start = state_loc[old_ind]
        i_stop = state_loc[old_ind] + (tm1_stop - tm1_start)#(tm1_stop - tm1_start)
        j_start = state_loc[ind]
        j_stop = state_loc[ind] + (t_stop - t_start)

        # q = (
        #     mess_fwd[tm1_start:tm1_stop].reshape(-1, 1)
        #     * T[aij, i_start:i_stop, j_start:j_stop]
        #     * mess_bwd[t_start:t_stop].reshape(1, -1)
        # )


        # (i_start, i_stop), (j_start, j_stop) = (
        #     state_loc[i : i + 2],
        #     state_loc[j : j + 2],
        # )

        # print("message shape: {}".format(np.shape(message)))

        # print("\n")
        # print("iteration {}".format(t))
        # print(tm1_start, tm1_stop, t_start, t_stop)
        # print(i_start, i_stop, j_start, j_stop)
        # print("T shape: {}".format(np.shape(T[aij, i_start:i_stop, j_start:j_stop])))
        message = np.ascontiguousarray(T[aij, i_start:i_stop, j_start:j_stop]).dot(
            message
        )
        # print("message shape: {}".format(np.shape(message)))
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        t_start, t_stop = mess_loc[t : t + 2]
        # print("mess loc shape: {}".format(mess_bwd[tm1_start:tm1_stop]))
        # print(t_start, t_stop)
        # print(message)
        mess_bwd[t_start:t_stop] = message
    return mess_bwd


# @nb.njit
def forward_mp(T_tr, Pi, n_clones, x, a, store_messages=False):
    """Log-probability of a sequence, and optionally, messages"""
    # pdb.set_trace()
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T_tr.dtype.type
    # pdb.set_trace()
    n_states = n_clones.sum()
    # print(n_states)
    Pi = np.ones(n_states) / n_states

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    j_start, j_stop = state_loc[j : j + 2]
    message = Pi[j_start:j_stop].copy().astype(dtype)
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_loc = np.hstack(
            (np.array([0], dtype=n_clones.dtype), n_clones[x])
        ).cumsum()
        mess_fwd = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    for t in range(1, x.shape[0]):
        # print(t)
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        new_message = np.zeros(j_stop - j_start, dtype=dtype)
        for d in range(len(new_message)):
            new_message[d] = (T_tr[aij, j_start + d, i_start:i_stop] * message).max()
        message = new_message
        p_obs = message.max()
        # print(p_obs)
        # print(message)
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd[t_start:t_stop] = message
    return log2_lik, mess_fwd


# @nb.njit
def rargmax(x):
    # return x.argmax()  # <- favors clustering towards smaller state numbers
    return np.random.choice((x == x.max()).nonzero()[0])


# @nb.njit
def backtrace(T, n_clones, x, a, mess_fwd):
    """Compute backward messages."""
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    code = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    t_start, t_stop = mess_loc[t : t + 2]
    belief = mess_fwd[t_start:t_stop]
    code[t] = rargmax(belief)
    ginis = []
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), j_start = state_loc[i : i + 2], state_loc[j]
        t_start, t_stop = mess_loc[t : t + 2]
        belief = (
            mess_fwd[t_start:t_stop] * T[aij, i_start:i_stop, j_start + code[t + 1]]
        )
        # JY added to calculate the distribution of beliefs:
        gini = gini_coefficient(belief)

        code[t] = rargmax(belief)
        ginis.append(gini)
    states = state_loc[x] + code
    return states, ginis


def backtraceE(T, E, n_clones, x, a, mess_fwd):
    """Compute backward messages."""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    states = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    belief = mess_fwd[t]
    states[t] = rargmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij = a[t]  # at time t -> t+1 we go from observation i to observation j
        belief = mess_fwd[t] * T[aij, :, states[t + 1]]
        states[t] = rargmax(belief)
    return states


def forwardE_mp(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """Log-probability of a sequence, and optionally, messages"""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = (T_tr[aij] * message.reshape(1, -1)).max(1)
        message *= E[:, j]
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def forward_mp_all(T_tr, Pi_x, Pi_a, n_clones, target_state, max_steps):
    """Log-probability of a sequence, and optionally, messages"""
    # forward pass
    t, log2_lik = 0, []
    message = Pi_x
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik.append(np.log2(p_obs))
    mess_fwd = []
    mess_fwd.append(message)
    T_tr_maxa = (T_tr * Pi_a.reshape(-1, 1, 1)).max(0)
    for t in range(1, max_steps):
        message = (T_tr_maxa * message.reshape(1, -1)).max(1)
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik.append(np.log2(p_obs))
        mess_fwd.append(message)
        if message[target_state] > 0:
            break
    else:
        assert False, "Unable to find a bridging path"
    return np.array(log2_lik), np.array(mess_fwd)


def backtrace_all(T, Pi_a, n_clones, mess_fwd, target_state):
    """Compute backward messages."""
    states = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    actions = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    n_states = T.shape[1]
    # backward pass
    t = mess_fwd.shape[0] - 1
    actions[t], states[t] = (
        -1,
        target_state,
    )  # last actions is irrelevant, use an invalid value
    for t in range(mess_fwd.shape[0] - 2, -1, -1):
        belief = (
            mess_fwd[t].reshape(1, -1) * T[:, :, states[t + 1]] * Pi_a.reshape(-1, 1)
        )
        a_s = rargmax(belief.flatten())
        actions[t], states[t] = a_s // n_states, a_s % n_states
    return actions, states

# @nb.njit
def gini_coefficient(values):
    """
    Calculate the Gini coefficient of a numpy array.

    Parameters:
    - values: a list or numpy array of values.

    Returns:
    - The Gini coefficient as a float.
    """
    # First we sort the array because the Gini coefficient formula assumes the array is sorted
    values = np.sort(values)
    # Calculate the cumulative sum of the sorted array
    cumsum = np.cumsum(values)
    # Calculate the Gini coefficient using the alternative formula
    n = len(values)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return gini