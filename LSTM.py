import numpy as np
np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def initialise_parameters(P, num_cells, features):
    # weights corresponing to new input, size = (feature, num_cells)
    P['Wi'], P['Wf'], P['Wo'] = (np.random.randn(3, num_cells, features) * 0.1) + 0.5
    P['Wz'] = np.random.randn(num_cells, features) * 0.1
    # weights corresponding to recurrence, size = (num_cells, num_cells)
    P['Ri'], P['Rf'], P['Ro'] = (np.random.randn(3, num_cells, num_cells) * 0.1) + 0.5
    P['Rz'] = np.random.randn(num_cells, num_cells) * 0.1
    # weights corresponding to bias, size = (1, num_cells)
    P['bz'], P['bi'], P['bf'], P['bo'] = np.random.randn(4, num_cells, 1) * 0.1
    # W weights, size  = (num_cells, features)
    P['Wv'] = np.random.rand(features, num_cells)*0.1
    # bias, size = (1, num_cells)
    P['bv'] = np.random.rand(features, 1)*0.1
    return P

def initialise_derivatives(num_cells, features):
    dP = {}
    # weights corresponing to new input, size = (feature, num_cells)
    dP['Wz'], dP['Wi'], dP['Wf'], dP['Wo'] = np.zeros((4, num_cells, features))
    # weights corresponding to recurrence, size = (num_cells, num_cells)
    dP['Rz'], dP['Ri'], dP['Rf'], dP['Ro'] = np.zeros((4, num_cells, num_cells))
    # weights corresponding to bias, size = (1, num_cells)
    dP['bz'], dP['bi'], dP['bf'], dP['bo'] = np.zeros((4, num_cells, 1))
    # W weights, size  = (num_cells, features)
    dP['Wv'] = np.zeros((features, num_cells))
    # bias, size = (1, num_cells)
    dP['bv'] = np.zeros((features, 1))
    return dP

# FOR USE WITH ADAM OPTIMIZER
def initialise_momentum(dP):
    M1, M2 = {}, {}
    for key in P:
        M1[key] = np.zeros_like(P[key])
        M2[key] = np.zeros_like(P[key])
    return M1, M2

def forward_pass_through_LSTM(x, state, P):
    h_prev, c_prev = state
    # forget gate
    f = sigmoid(np.dot(P['Wf'], x) + np.dot(P['Rf'], h_prev) + P['bf'])
    # input gate
    i = sigmoid(np.dot(P['Wi'], x) + np.dot(P['Ri'], h_prev) + P['bi'])
    # z neuron
    z = tanh(np.dot(P['Wz'], x) + np.dot(P['Rz'], h_prev) + P['bz'])
    # cell state
    c = f*c_prev + i*z
    # output gate
    o = sigmoid(np.dot(P['Wo'], x) + np.dot(P['Ro'], h_prev) + P['bo'])
    # hidden state
    h = o * tanh(c)

    cache = f, i, z, o
    state = h, c
    return state, cache

def forward_pass_through_dense(x, P):
    # Dense layer with softmax activation
    v = np.dot(P['Wv'], x) + P['bv']
    y = stable_softmax(v)
    return y

def backprop_through_dense(x, y, target, dP):
    # computer error
    dv = np.copy(y)
    dv[np.argmax(target)] -= 1
    dP['Wv'] += np.dot(dv, x.T)
    dP['bv'] += dv
    return dv, dP

def backward_pass_through_LSTM(deriv, passing_state, x, state_prev, state, cache, P, dP):
    # unpack cache and state
    f, i, z, o = cache
    h, c = state
    h_prev, c_prev = state_prev
    dh_next, dc_next = passing_state
    # hidden state
    dh = np.dot(P['Wv'].T, deriv)
    dh += dh_next

    # open gate
    do = dsigmoid(o) * dh * tanh(c)
    dP['Wo'] += np.dot(do, x.T)
    dP['Ro'] += np.dot(do, h_prev.T)
    dP['bo'] += do

    # cell state
    dc = np.copy(dc_next)
    dc += dh * o * dtanh(tanh(c))

    # z neuron
    dz = dtanh(z) * dc * i
    dP['Wz'] += np.dot(dz, x.T)
    dP['Rz'] += np.dot(dz, h_prev.T)
    dP['bz'] += dz

    # input gate
    di = dsigmoid(i) * dc * z
    dP['Wi'] += np.dot(di, x.T)
    dP['Ri'] += np.dot(di, h_prev.T)
    dP['bi'] += di

    # forget gate
    df = dsigmoid(f) * dc * c_prev
    dP['Wf'] += np.dot(df, x.T)
    dP['Rf'] += np.dot(df, h_prev.T)
    dP['bf'] += df

    # if passing to lower layer
    # dx = (np.dot(P['Wf'].T, df) + np.dot(P['Wi'].T, di) + np.dot(P['Wz'].T, dz) + np.dot(P['Wo'].T, do))

    # pass hidden state
    dh_prev = (np.dot(P['Rf'].T, df) + np.dot(P['Ri'].T, di) + np.dot(P['Rz'].T, dz) + np.dot(P['Ro'].T, do))

    # pass cell state
    dc_prev = f * dc
    passing_state = dh_prev, dc_prev
    return passing_state, dP

def forward_backward_pass(inputs, targets2, input_state, P, M1, M2, optimizer, learning_rate, t):
    x, y, cache_1, state = {}, {}, {}, {}
    state[-1] = (np.copy(input_state[0]), np.copy(input_state[1]))
    loss = 0

    # Forward pass through time steps
    for t in range(len(inputs)):
        (state[t], cache_1[t]) = forward_pass_through_LSTM(np.array(inputs[t]).reshape(-1,1), state[t-1], P)
        (y[t]) = forward_pass_through_dense(state[t][0], P)

        # cross entropy loss
        loss += -np.log(y[t][np.argmax(targets2[t]), 0])

    # Backward pass through time steps
    passing_state = (np.zeros_like(state[-1][0]), np.zeros_like(state[-1][1]))
    dP = initialise_derivatives(num_cells, features) # clear gradients
    for t in reversed(range(len(inputs))):
        dv, dP = backprop_through_dense(state[t][0], y[t], targets2[t], dP)
        passing_state, dP = backward_pass_through_LSTM(dv, passing_state, np.array(inputs[t]).reshape(-1,1), state[t-1], state[t], cache_1[t], P, dP)

    P, M1, M2, t = update_parameters(optimizer, P, dP, M1, M2, t, learning_rate = learning_rate)
    return loss, state[len(inputs)-1], P, M1, M2, t

def prediction(state, x, length, P, alphabet):
    x = np.array(x).reshape(-1,1)
    pred = ''
    for t in range(length):
        state, _ = forward_pass_through_LSTM(x, state, P)
        p = forward_pass_through_dense(state[0], P)
        idx = np.random.choice(range(features), p=p.ravel())
        x = np.zeros((features, 1))
        x[idx] = 1
        pred += alphabet[idx]
    return pred

def update_parameters(optimizer, P, dP, M1, M2, t,  learning_rate = 0.001, decay_rate_1 = 0.9, decay_rate_2 = 0.999):
    if optimizer == 'adam':
        for key in P:
            t = t + 1
            dP[key] = np.clip(dP[key], -5, 5, out= dP[key])
            M1[key] = (decay_rate_1)*M1[key] + (1 - decay_rate_1)*dP[key]
            M2[key] = (decay_rate_2)*M2[key] + (1 - decay_rate_2)*(dP[key]**2)
            m1 = M1[key]/(1 - decay_rate_1**(t))
            m2 = M2[key]/(1 - decay_rate_2**(t))

            P[key] = P[key] - learning_rate*(m1/(np.sqrt(m2)+ 1e-8))
    else:
        for key in P:
            dP[key] = np.clip(dP[key], -1, 1, out= dP[key])
            P[key] -= learning_rate * dP[key]

    return P, M1, M2, t



################################################################################


data = 'stevenstevenstevenstevenstevenstevenstevenstevensteven'
optimizer = 'adam'  # 'adam' or 'GD'
num_cells = 100
T_steps = 4
num_iterations = 100


# One hot encode
alphabet = sorted(list(set(data)))
features = len(alphabet)
encoded_data = [[0 if char != letter else 1 for char in alphabet] for letter in data]

state = (np.zeros((num_cells, 1)), np.zeros((num_cells, 1)))

# Initialise parameters
P = {}
P = initialise_parameters(P, num_cells, features)
M1, M2 = initialise_momentum(P)

iteration, location = 0, 0
Loss = []
reset = False

if optimizer == 'adam':
    learning_rate = 0.001
    t = 0
elif optimizer == 'GD':
    learning_rate = 2
    t = 0
else:
    print('Unrecognised optimizer, default set to adam')
    optimizer == 'adam'

while iteration < num_iterations:
    if location + T_steps >= len(data):
        reset = True
        # last chunk of data
        inputs = encoded_data[location: -1]
        targets = encoded_data[location + 1:]
    else:
        # define inputs and outputs
        inputs = encoded_data[location: location + T_steps]
        targets = encoded_data[location + 1: location + T_steps + 1]


    loss, state, P, M1, M2, t = forward_backward_pass(inputs, targets, state, P, M1, M2, optimizer, learning_rate, t)
    Loss.append(loss)

    # Print an example prediction every 500 iterations
    if (iteration % 500 == 0) and (iteration != 0) and reset:
        if optimizer == 'GD':
            learning_rate /= 2
        print(prediction(state, encoded_data[0], 40, P, alphabet))

    if reset:
        reset = False
        state = (np.zeros((num_cells, 1)), np.zeros((num_cells, 1)))
        location = 0
        iteration += 1
    else:
        location += T_steps

# Pass through with no backprop (also passes through last element of training set, unlike during training)
state = (np.zeros((num_cells, 1)), np.zeros((num_cells, 1)))
for t in range(len(encoded_data)):
    (state, _) = forward_pass_through_LSTM(np.array(encoded_data[t]).reshape(-1,1), state, P)
    _ = forward_pass_through_dense(state[0], P)

print(prediction(state, encoded_data[0], 40, P, alphabet))
