import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import models.config as config


def generate_obstacle_circle(center, radius, num=12):
    theta = np.linspace(0, np.pi*2, num=num, endpoint=False).reshape(-1, 1)
    unit_circle = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    circle = np.array(center) + unit_circle * radius
    return circle


def generate_obstacle_rectangle(center, sides, num=12):
    # calculate the number of points on each side of the rectangle
    a, b = sides # side lengths
    n_side_1 = int(num // 2 * a / (a+b))
    n_side_2 = num // 2 - n_side_1
    n_side_3 = n_side_1
    n_side_4 = num - n_side_1 - n_side_2 - n_side_3
    # top
    side_1 = np.concatenate([
        np.linspace(-a/2, a/2, n_side_1, endpoint=False).reshape(-1, 1), 
        b/2 * np.ones(n_side_1).reshape(-1, 1)], axis=1)
    # right
    side_2 = np.concatenate([
        a/2 * np.ones(n_side_2).reshape(-1, 1),
        np.linspace(b/2, -b/2, n_side_2, endpoint=False).reshape(-1, 1)], axis=1)
    # bottom
    side_3 = np.concatenate([
        np.linspace(a/2, -a/2, n_side_3, endpoint=False).reshape(-1, 1), 
        -b/2 * np.ones(n_side_3).reshape(-1, 1)], axis=1)
    # left
    side_4 = np.concatenate([
        -a/2 * np.ones(n_side_4).reshape(-1, 1),
        np.linspace(-b/2, b/2, n_side_4, endpoint=False).reshape(-1, 1)], axis=1)

    rectangle = np.concatenate([side_1, side_2, side_3, side_4], axis=0)
    rectangle = rectangle + np.array(center)
    return rectangle


# YY: Generate start and goal points
def generate_data(num_agents, dist_min_thres):
    side_length = np.sqrt(max(1.0, num_agents / 8.0))
    states = np.zeros(shape=(num_agents + 1, 2), dtype=np.float32)
    goals = np.zeros(shape=(num_agents + 1, 2), dtype=np.float32)

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(size=(2,)) * side_length
        dist_min = np.linalg.norm(states - candidate, axis=1).min()
        if dist_min <= dist_min_thres:
            continue
        states[i] = candidate
        i = i + 1

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i]
        dist_min = np.linalg.norm(goals - candidate, axis=1).min()
        if dist_min <= dist_min_thres:
            continue
        goals[i] = candidate
        i = i + 1

    # YY: Add our agent's state & goal
    states[-1] = np.array([0, 0], dtype=np.float32).reshape((2,))
    goals[-1] = np.array([side_length, side_length], dtype=np.float32).reshape((2,))

    # print("states: ", states)
    # print("goals: ", goals)
    states = np.concatenate(
        [states, np.zeros(shape=(num_agents + 1, 2), dtype=np.float32)], axis=1)
    # print("conc_states: ", states)
    return states, goals
















def network_cbf(x, r):
    """
    Description:
        CBF Neural Netwrok

    Args:
        x: [topk + 1, topk + 1, 4] matrix
        r: threshold

    Returns:
        h: CBF value h(agent, obstacle). Shape: [topk + 1, 1].
            e.g., h = [h(agent, obstacle1), h(agent, obstacle2), ..., h(agent, agent)]
    """
    # Get 2 norm distance between each agent, shape: [13, 13]
    d_norm = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :2]) + 1e-4, axis=2))

    # Add dim. Shape: [13, 13, 6]
    x = tf.concat([x,
        tf.expand_dims(tf.eye(tf.shape(x)[0]), 2),
        tf.expand_dims(d_norm - r, 2)], axis=2)

    # Compute distance matrix. Shape: [13, 13, 1]
    dist = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :2]) + 1e-4, axis=2, keepdims=True))

    # If distance is too large, the obs will be overlook, shape: [13, 13, 1]
    mask = tf.cast(tf.less_equal(dist, config.OBS_RADIUS), tf.float32)

    # Only take the last row which is the agent line
    x = x[-1:, :, :]  # shape: [1, 13, 6]. The last element in dim1 is 0 because it's agent-agent
    mask = mask[-1:, :, :]  # shape: [1, 13, 1]


    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_1',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=128,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_2',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_3',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=1,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_4',
                                 activation_fn=None)
    # x shape: [1, 13, 1]
    h = x * mask  # filter out too far away obs
    return h, mask


def dynamics(s, a):
    """
    Description:
        The ground robot dynamics.
    
    Args:
        s (N, 4): The current state.
        a (N, 2): The acceleration taken by each agent.

    Returns:
        dsdt (N, 4): The time derivative of s.
    """
    dsdt = tf.concat([s[:, 2:], a], axis=1)
    return dsdt


def loss_barrier(h, dang_mask_reshape, safe_mask_reshape, eps=[1e-3, 0]):
    """
    Description:
        Build the loss function for the control barrier functions.

    Args:
        h (1, topk + 1, 1): The control barrier function.
        idx: index of the dangerous state. -1 if no dangerous state.
        eps: threshold

    Returns:
        loss_dang, loss_safe, acc_dang, acc_safe
    """

    # h shape to [topk + 1,]
    h_reshape = tf.reshape(h, [-1])

    # Form dangerous and safe h
    dang_h = tf.boolean_mask(h_reshape, dang_mask_reshape)
    safe_h = tf.boolean_mask(h_reshape, safe_mask_reshape)

    # Get number of dangerous and safe h
    num_dang = tf.cast(tf.shape(dang_h)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_h)[0], tf.float32)

    # Calculate dangerous and safe loss
    loss_dang = tf.reduce_sum(
        tf.math.maximum(dang_h + eps[0], 0)) / (1e-5 + num_dang)
    loss_safe = tf.reduce_sum(
        tf.math.maximum(-safe_h + eps[1], 0)) / (1e-5 + num_safe)

    # Calculate dangerous and safe accuracy
    acc_dang = tf.reduce_sum(tf.cast(
        tf.less_equal(dang_h, 0), tf.float32)) / (1e-5 + num_dang)
    acc_safe = tf.reduce_sum(tf.cast(
        tf.greater(safe_h, 0), tf.float32)) / (1e-5 + num_safe)
    acc_dang = tf.cond(
        tf.greater(num_dang, 0), lambda: acc_dang, lambda: -tf.constant(1.0))
    acc_safe = tf.cond(
        tf.greater(num_safe, 0), lambda: acc_safe, lambda: -tf.constant(1.0))

    return loss_dang, loss_safe, acc_dang, acc_safe


def loss_derivatives(s, a, h, s_next, alpha, r, eps=[1e-3, 0]):
    """
    Description:
        Calculate the derivative loss.

    Args:
        s (topk + 1, 4): current timestep state
        a (2,): current timestep action of the agent
        h (topk + 1, 1): the control barrier function
        s_next (topk + 1, 4): next timestep state
        alpha: alpha CBF
        r: threshold
        eps: threshold

    Returns:
        loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
    """

    # Get current timestep state of the agent
    s_agent = s[-1:, :]  # [1, 4]

    # Calculate next timestep state of the agent
    dsdt = dynamics(s_agent, a)
    # TODO: Maybe we only need to ensure the agent will not enter unsafe states,
    #       which means we don't need to concatenate with other obstacles to calculate the whole vector of derivative.
    s_agent_next = s_agent + dsdt * config.TIME_STEP

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:-1, :]  # [topk, 4]

    # Combine next timestep state of the agent and obstacle to get the next timestep state
    s_next_real = tf.concat([s_obstacles_next, s_agent_next], axis=0)

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 1) - tf.expand_dims(s_next_real, 0)
    h_next, _ = network_cbf(x=x_next, r=r)

    # Calculate h derivative
    # TODO: Here's wrong!!! you should calculate deriv by min(h(s_next)) - min(h(s))
    deriv = h_next - h + config.TIME_STEP * alpha * h  # deriv h
    deriv_reshape = tf.reshape(deriv, [-1])[:-1]  # [topk + 1, ]  # add [:-1] because don't count the last element in

    # Form dang_deriv and safe_deriv mask via h
    safe_deriv_mask = tf.reshape(tf.greater_equal(h, 0), [-1])[:-1]  # add [:-1] because don't count the last element in
    dang_deriv_mask = tf.logical_not(safe_deriv_mask)

    # Calculate safe and dangerous derivative loss
    dang_deriv = tf.boolean_mask(deriv_reshape, dang_deriv_mask)
    safe_deriv = tf.boolean_mask(deriv_reshape, safe_deriv_mask)
    num_dang = tf.cast(tf.shape(dang_deriv)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_deriv)[0], tf.float32)
    loss_dang_deriv = tf.reduce_sum(
        tf.math.maximum(-dang_deriv + eps[0], 0)) / (1e-5 + num_dang)
    loss_safe_deriv = tf.reduce_sum(
        tf.math.maximum(-safe_deriv + eps[1], 0)) / (1e-5 + num_safe)

    # Calculate safe and dangerous derivative accuracy
    acc_dang_deriv = tf.reduce_sum(tf.cast(
        tf.greater_equal(dang_deriv, 0), tf.float32)) / (1e-5 + num_dang)
    acc_safe_deriv = tf.reduce_sum(tf.cast(
        tf.greater_equal(safe_deriv, 0), tf.float32)) / (1e-5 + num_safe)
    acc_dang_deriv = tf.cond(
        tf.greater(num_dang, 0), lambda: acc_dang_deriv, lambda: -tf.constant(1.0))
    acc_safe_deriv = tf.cond(
        tf.greater(num_safe, 0), lambda: acc_safe_deriv, lambda: -tf.constant(1.0))

    # Return unsafe states
    unsafe_states = tf.cond(tf.greater(num_dang, 0), lambda: tf.constant(1.0), lambda: -tf.constant(1.0))
        # tf.cond(tf.greater(num_dang, 0), lambda: tf.boolean_mask(s, dang_deriv_mask), lambda: -tf.constant(1.0))

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv, num_dang, num_safe, unsafe_states


def loss_derivatives_min(s, a, h, s_next, alpha, r, eps=[1e-3, 0]):
    """
    Description:
        Calculate the derivative loss.

    Args:
        s (topk + 1, 4): current timestep state
        a (2,): current timestep action of the agent
        h (topk + 1, 1): the control barrier function
        s_next (topk + 1, 4): next timestep state
        alpha: alpha CBF
        r: threshold
        eps: threshold

    Returns:
        loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
    """

    # todo: if curr and next timestep are both in safe region, return loss = 0

    # Get current timestep state of the agent
    s_agent = s[-1:, :]  # [1, 4]

    # Calculate next timestep state of the agent
    dsdt = dynamics(s_agent, a)

    s_agent_next = s_agent + dsdt * config.TIME_STEP

    # yy: one more step for the agent
    #  -> s_agent = [pos_n_1 + delta_time * v_n_1, pos_n_2 + delta_time * v_n_2, v_n_1, v_n_2]
    # s_agent_next_next = \
    #     tf.concat([s_agent_next[-1:, :2] + s_agent_next[-1:, 2:] * config.TIME_STEP, s_agent_next[-1:, 2:]], axis=1)

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:-1, :]  # [topk, 4]

    # yy: one more step for obstacles
    # s_obstacles_next_next = \
    #     tf.concat([s_obstacles_next[:, :2] + s_obstacles_next[:, 2:] * config.TIME_STEP, s_obstacles_next[:, 2:]], axis=1)

    # yy: Combine next 2 timesteps state of the agent and obstacle to get the next timestep state
    # s_next_real = tf.concat([s_obstacles_next_next, s_agent_next_next], axis=0)

    s_next_real = tf.concat([s_obstacles_next, s_agent_next], axis=0)

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 1) - tf.expand_dims(s_next_real, 0)
    h_next, _ = network_cbf(x=x_next, r=r)

    # Calculate h derivative
    # yy: use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:, :-1, :])
    h_next_min = tf.math.reduce_min(h_next[:, :-1, :])
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min
    deriv_reshape = tf.reshape(deriv, [-1])
    loss = tf.cond(
        tf.greater(h_min, 0),
        lambda: tf.math.maximum(-deriv_reshape + eps[1], 0),
        lambda: tf.math.maximum(-deriv_reshape + eps[0], 0) * 2  # yy: if it's unsafe state, add weight (e.g., 2) to the loss
    )

    # # yy: if two timesteps the h are both > 0, then don't calculate the gradient
    # loss = tf.cond(
    #     tf.math.logical_and(tf.greater(h_min, 0), tf.greater(h_next_min, 0)),
    #     lambda: tf.constant(0),
    #     lambda: loss
    # )

    acc = tf.reduce_sum(tf.cast(tf.greater_equal(deriv_reshape, 0), tf.float32))

    return loss, acc, h_min, h_next_min, deriv_reshape




















# YY: this is the goal reaching loss
def loss_actions(s, g, a, r, ttc):
    state_gain = -tf.constant(
        np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=tf.float32)
    s_ref = tf.concat([s[:, :2] - g, s[:, 2:]], axis=1)
    action_ref = tf.linalg.matmul(s_ref, state_gain, False, True)
    action_ref_norm = tf.reduce_sum(tf.square(action_ref), axis=1)
    action_net_norm = tf.reduce_sum(tf.square(a), axis=1)
    norm_diff = tf.abs(action_net_norm - action_ref_norm)
    loss = tf.reduce_mean(norm_diff)
    return loss


def statics(s, a, h, alpha, indices=None):
    dsdt = dynamics(s, a)
    s_next = s + dsdt * config.TIME_STEP

    x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
    h_next, mask_next, _ = network_cbf(x=x_next, r=config.DIST_MIN_THRES, indices=indices)

    deriv = h_next - h + config.TIME_STEP * alpha * h

    mean_deriv = tf.reduce_mean(deriv)
    std_deriv = tf.sqrt(tf.reduce_mean(tf.square(deriv - mean_deriv)))
    prob_neg = tf.reduce_mean(tf.cast(tf.less(deriv, 0), tf.float32))

    return mean_deriv, std_deriv, prob_neg


def ttc_dangerous_mask(s, r, ttc, indices=None):
    s_diff = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    s_diff = tf.concat(
        [s_diff, tf.expand_dims(tf.eye(tf.shape(s)[0]), 2)], axis=2)
    s_diff, _ = remove_distant_agents(s_diff, config.TOP_K, indices)
    x, y, vx, vy, eye = tf.split(s_diff, 5, axis=2)
    x = x + eye
    y = y + eye
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2
    dist_dangerous = tf.less(gamma, 0)

    has_two_positive_roots = tf.logical_and(
        tf.greater(beta ** 2 - 4 * alpha * gamma, 0),
        tf.logical_and(tf.greater(gamma, 0), tf.less(beta, 0)))
    root_less_than_ttc = tf.logical_or(
        tf.less(-beta - 2 * alpha * ttc, 0),
        tf.less((beta + 2 * alpha * ttc) ** 2, beta ** 2 - 4 * alpha * gamma))
    has_root_less_than_ttc = tf.logical_and(has_two_positive_roots, root_less_than_ttc)
    ttc_dangerous = tf.logical_or(dist_dangerous, has_root_less_than_ttc)

    return ttc_dangerous


def ttc_dangerous_mask_np(s, r, ttc):
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    # s_diff[:-1, :, :2] = float('inf')  # YY: only consider our agent
    x, y, vx, vy = np.split(s_diff, 4, axis=2)
    x = x + np.expand_dims(np.eye(np.shape(s)[0]), 2)
    y = y + np.expand_dims(np.eye(np.shape(s)[0]), 2)
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2
    dist_dangerous = np.less(gamma, 0)

    has_two_positive_roots = np.logical_and(
        np.greater(beta ** 2 - 4 * alpha * gamma, 0),
        np.logical_and(np.greater(gamma, 0), np.less(beta, 0)))
    root_less_than_ttc = np.logical_or(
        np.less(-beta - 2 * alpha * ttc, 0),
        np.less((beta + 2 * alpha * ttc) ** 2, beta ** 2 - 4 * alpha * gamma))
    has_root_less_than_ttc = np.logical_and(has_two_positive_roots, root_less_than_ttc)
    ttc_dangerous = np.logical_or(dist_dangerous, has_root_less_than_ttc)

    return ttc_dangerous


# YY: indices should be the rest agents (?)
def remove_distant_agents(x, k, indices=None):
    n, _, c = x.get_shape().as_list()
    if n <= k:   # if #agents < topk, return directly
        return x, False
    d_norm = tf.sqrt(tf.reduce_sum(tf.square(x[:, :, :2]) + 1e-6, axis=2))
    if indices is not None:
        x = tf.reshape(tf.gather_nd(x, indices), [n, k, c])
        return x, indices
    _, indices = tf.nn.top_k(-d_norm, k=k)  # shape of d_norm: [17, 17]. return topk indices in dimension 2 in shape: [17, 12]
    row_indices = tf.expand_dims(
        tf.range(tf.shape(indices)[0]), 1) * tf.ones_like(indices)
    row_indices = tf.reshape(row_indices, [-1, 1])
    column_indices = tf.reshape(indices, [-1, 1])
    # change original indices to [17 * 12, 2]. e.g., [..., ..., [16, 4], [16, 8], [16, 1], ...]
    indices = tf.concat([row_indices, column_indices], axis=1)
    x = tf.reshape(tf.gather_nd(x, indices), [n, k, c])
    return x, indices


# yy: batch based -------------------


# yy: CBF NN only consider pos info
def network_cbf_vel(x, r):
    """
    Description:
        CBF Neural Netwrok

    Args:
        x: [topk + 1, topk + 1, 2] matrix
        r: threshold

    Returns:
        h: CBF value h(agent, obstacle). Shape: [topk + 1, 1].
            e.g., h = [h(agent, obstacle1), h(agent, obstacle2), ..., h(agent, agent)]
    """
    # Get 2 norm distance between each agent, shape: [13, 13]
    d_norm = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :2]) + 1e-4, axis=2))

    # Add dim. Shape: [13, 13, 4]
    x = tf.concat([x,
        tf.expand_dims(tf.eye(tf.shape(x)[0]), 2),
        tf.expand_dims(d_norm - r, 2)], axis=2)

    # Compute distance matrix. Shape: [13, 13, 1]
    dist = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :2]) + 1e-4, axis=2, keepdims=True))

    # If distance is too large, the obs will be overlook, shape: [13, 13, 1]
    mask = tf.cast(tf.less_equal(dist, config.OBS_RADIUS), tf.float32)

    # Only take the last row which is the agent line
    x = x[-1:, :, :]  # shape: [1, 13, 4]. The last element in dim1 is 0 because it's agent-agent
    mask = mask[-1:, :, :]  # shape: [1, 13, 1]


    # x = tf.contrib.layers.conv1d(inputs=x,
    #                              num_outputs=64,
    #                              kernel_size=1,
    #                              reuse=tf.AUTO_REUSE,
    #                              scope='cbf/conv_1',
    #                              activation_fn=tf.nn.relu)
    # x = tf.contrib.layers.conv1d(inputs=x,
    #                              num_outputs=128,
    #                              kernel_size=1,
    #                              reuse=tf.AUTO_REUSE,
    #                              scope='cbf/conv_2',
    #                              activation_fn=tf.nn.relu)
    # x = tf.contrib.layers.conv1d(inputs=x,
    #                              num_outputs=64,
    #                              kernel_size=1,
    #                              reuse=tf.AUTO_REUSE,
    #                              scope='cbf/conv_3',
    #                              activation_fn=tf.nn.relu)
    # x = tf.contrib.layers.conv1d(inputs=x,
    #                              num_outputs=1,
    #                              kernel_size=1,
    #                              reuse=tf.AUTO_REUSE,
    #                              scope='cbf/conv_4',
    #                              activation_fn=None)

    x = tf.layers.conv1d(inputs=x,
                                 filters=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 name='cbf/conv_1',
                                 activation=tf.nn.relu)
    x = tf.layers.conv1d(inputs=x,
                                 filters=128,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 name='cbf/conv_2',
                                 activation=tf.nn.relu)
    x = tf.layers.conv1d(inputs=x,
                                 filters=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 name='cbf/conv_3',
                                 activation=tf.nn.relu)
    x = tf.layers.conv1d(inputs=x,
                                 filters=1,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 name='cbf/conv_4',
                                 activation=None)

    # x shape: [1, 13, 1]
    h = x * mask  # filter out too far away obs
    return h, mask




def network_cbf_batch(x, r):
    """
    Description:
        CBF Neural Netwrok

    Args:
        x: [bs, topk + 1, topk + 1, 2] matrix
        r: threshold

    Returns:
        h: CBF value h(agent, obstacle). Shape: [topk + 1, 1].
            e.g., h = [h(agent, obstacle1), h(agent, obstacle2), ..., h(agent, agent)]
    """
    # Get 2 norm distance between each agent, shape: [bs, 13, 13]
    d_norm = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :, :2]) + 1e-4, axis=3))

    # Add dim. Shape: [bs, 13, 13, 4]
    x = tf.concat([x,
        tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(tf.shape(x)[1]), 2), 0), [tf.shape(x)[0], 1, 1, 1]),
        tf.expand_dims(d_norm - r, 3)], axis=3)

    # Compute distance matrix. Shape: [bs, 13, 13, 1]
    dist = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :, :2]) + 1e-4, axis=3, keepdims=True))

    # If distance is too large, the obs will be overlook, shape: [bs, 13, 13, 1]
    mask = tf.cast(tf.less_equal(dist, config.OBS_RADIUS), tf.float32)

    # Only take the last row which is the agent line
    x = x[:, -1, :, :]  # shape: [bs, 13, 4]. The last element in dim1 is 0 because it's agent-agent
    mask = mask[:, -1, :, :]  # shape: [bs, 13, 1]


    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_1',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=128,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_2',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_3',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=1,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_4',
                                 activation_fn=None)
    # x shape: [bs, 13, 1]
    h = x * mask  # filter out too far away obs
    return h, mask


# yy: loss_deriv
def loss_derivatives_min_batch(s, a, h, s_next, alpha, r, eps=[1e-3, 0]):
    """
    Description:
        Calculate the derivative loss.

    Args:
        s (bs, topk + 1, 4): current timestep state
        a (bs, 1, 2): current timestep action of the agent
        h (bs, topk + 1): the control barrier function
        s_next (bs, topk + 1, 4): next timestep state
        alpha: alpha CBF
        r: threshold
        eps: threshold

    Returns:
        loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
    """

    # todo: if curr and next timestep are both in safe region, return loss = 0

    bs = s.get_shape().as_list()[0]

    # Get current timestep state of the agent
    s_agent = s[:, -1:, :]  # [bs, 1, 4]

    # Calculate next timestep state of the agent
    dsdt = tf.concat([s_agent[:, :, 2:], a], axis=2)
    s_agent_next = s_agent + dsdt * config.TIME_STEP  # [bs, 1, 4] [px + vx, py + vy, vx + ax, vy + ay]
    # get 2nd timestep's pos
    s_agent_next = s_agent_next[:, :, :2] + s_agent_next[:, :, 2:] * config.TIME_STEP  # [bs, 1, 2] [px + 2vx + ax, py + 2vy + ay]

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:, :-1, :]  # [bs, topk, 4] [px, py, vx, vy]
    # get 2nd timestep's pos
    s_obstacles_next = s_obstacles_next[:, :, :2] + s_obstacles_next[:, :, 2:] * config.TIME_STEP  # [bs, topk, 2]


    # The next state using combination
    s_next_real = tf.concat([s_obstacles_next, s_agent_next], axis=1)  # [bs, topk + 1, 2]

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 2) - tf.expand_dims(s_next_real, 1)  # [bs, topk + 1, topk + 1, 2]
    h_next = tf.compat.v1.map_fn(fn=lambda t: tf.reshape(network_cbf_vel(x=t, r=r)[0], [-1]), elems=x_next)  # [bs, topk + 1]

    # Calculate h derivative
    # yy: use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:, :-1], axis=1)  # [bs,]
    h_next_min = tf.math.reduce_min(h_next[:, :-1], axis=1)  # [bs,]
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min  # [bs,]
    deriv_reshape = tf.reshape(deriv, [-1])  # [bs,]

    # h>0 mask and h<0 mask
    safe_mask = tf.greater_equal(h_min, 0)
    dang_mask = tf.less(h_min, 0)
    # bs = a + b. deriv_safe: [a], deriv_dang: [b]
    deriv_safe = tf.boolean_mask(deriv_reshape, safe_mask)
    # deriv_safe = tf.constant(0.0, dtype=tf.float32)
    deriv_dang = tf.boolean_mask(deriv_reshape, dang_mask)
    loss_safe_deriv = tf.math.maximum(-deriv_safe + eps[1], 0)  # [a]
    loss_dang_deriv = tf.math.maximum(-deriv_dang + eps[0], 0) * 2  # [b]
    loss_ls = tf.concat([loss_safe_deriv, loss_dang_deriv], axis=0)
    # loss_ls = tf.concat([loss_safe_deriv], axis=0)  # yy: only calculate the safe deriv
    # loss_ls = tf.concat([loss_dang_deriv], axis=0)  # yy: only calculate the dangerous deriv
    loss = tf.reduce_mean(loss_ls)

    acc = tf.reduce_mean(tf.cast(tf.greater_equal(deriv_reshape, 0), tf.float32))

    return loss, acc, h_min, h_next_min, deriv_reshape, deriv_safe, deriv_dang


# yy: loss_deriv without calculating 2 steps to get the deriv
def loss_derivatives_min_batch_original(s, a, h, s_next, alpha, r, eps=[1e-3, 0]):
    """
    Description:
        Calculate the derivative loss.

    Args:
        s (bs, topk + 1, 4): current timestep state
        a (bs, 1, 2): current timestep action of the agent
        h (bs, topk + 1): the control barrier function
        s_next (bs, topk + 1, 4): next timestep state
        alpha: alpha CBF
        r: threshold
        eps: threshold

    Returns:
        loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
    """

    # todo: if curr and next timestep are both in safe region, return loss = 0

    bs = s.get_shape().as_list()[0]

    # Get current timestep state of the agent
    s_agent = s[:, -1:, :]  # [bs, 1, 4]

    # Calculate next timestep state of the agent
    dsdt = tf.concat([s_agent[:, :, 2:], a], axis=2)
    s_agent_next = s_agent + dsdt * config.TIME_STEP  # [bs, 1, 4] [px + vx, py + vy, vx + ax, vy + ay]
    # # get 2nd timestep's pos
    # s_agent_next = s_agent_next[:, :, :2] + s_agent_next[:, :, 2:] * config.TIME_STEP  # [bs, 1, 2] [px + 2vx + ax, py + 2vy + ay]

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:, :-1, :]  # [bs, topk, 4] [px, py, vx, vy]
    # # get 2nd timestep's pos
    # s_obstacles_next = s_obstacles_next[:, :, :2] + s_obstacles_next[:, :, 2:] * config.TIME_STEP  # [bs, topk, 2]


    # The next state using combination
    s_next_real = tf.concat([s_obstacles_next, s_agent_next], axis=1)  # [bs, topk + 1, 4]

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 2) - tf.expand_dims(s_next_real, 1)  # [bs, topk + 1, topk + 1, 4]
    h_next = tf.compat.v1.map_fn(fn=lambda t: tf.reshape(network_cbf_vel(x=t, r=r)[0], [-1]), elems=x_next)  # [bs, topk + 1]

    # Calculate h derivative
    # yy: use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:, :-1], axis=1)  # [bs,]
    h_next_min = tf.math.reduce_min(h_next[:, :-1], axis=1)  # [bs,]
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min  # [bs,]
    deriv_reshape = tf.reshape(deriv, [-1])  # [bs,]

    # h>0 mask and h<0 mask
    safe_mask = tf.greater(h_min, 0)
    dang_mask = tf.less_equal(h_min, 0)
    # bs = a + b. deriv_safe: [a], deriv_dang: [b]
    deriv_safe = tf.boolean_mask(deriv_reshape, safe_mask)
    deriv_dang = tf.boolean_mask(deriv_reshape, dang_mask)
    loss_safe_deriv = tf.math.maximum(-deriv_safe + eps[1], 0)  # [a]
    loss_dang_deriv = tf.math.maximum(-deriv_dang + eps[0], 0) * 2  # [b]
    loss_ls = tf.concat([loss_safe_deriv, loss_dang_deriv], axis=0)
    loss = tf.reduce_mean(loss_ls)

    acc = tf.reduce_mean(tf.cast(tf.greater_equal(deriv_reshape, 0), tf.float32))

    return loss, acc, h_min, h_next_min, deriv_reshape, deriv_safe, deriv_dang


def calc_deriv(alpha, r):
    """
    Description:
        Calculate the derivative loss.

    Args:
        alpha: alpha CBF

    Returns:
        is_safe_action: bool
    """
    # placeholders
    s = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='state')
    s_next = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='next_state')
    a = tf.placeholder(tf.float32, [1, 2], name='action')
    ph = [s, s_next, a]

    # Get current timestep state of the agent
    s_agent = s[-1:, :]  # [1, 4]

    # Calculate next timestep state of the agent
    dsdt = tf.concat([s_agent[:, 2:], a], axis=1)
    s_agent_next = s_agent + dsdt * config.TIME_STEP  # [1, 4]

    # one more step for agent
    s_agent_2nd = s_agent_next[:, :2] + s_agent_next[:, 2:] * config.TIME_STEP

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:-1, :]  # [topk, 4]

    # one more step for obs
    s_obstacles_2nd = s_obstacles_next[:, :2] + s_obstacles_next[:, 2:] * config.TIME_STEP

    # The next state using combination
    s_next_real = tf.concat([s_obstacles_2nd, s_agent_2nd], axis=0)  # [topk + 1, 2]

    # Get h of current timestep
    x = tf.expand_dims(s[:, :2], 1) - tf.expand_dims(s[:, :2], 0)  # [topk + 1, topk + 1, 2]
    h = tf.reshape(network_cbf_vel(x=x, r=r)[0], [-1])

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 1) - tf.expand_dims(s_next_real, 0)  # [topk + 1, topk + 1, 2]
    h_next = tf.reshape(network_cbf_vel(x=x_next, r=r)[0], [-1])

    # Calculate h derivative, use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:-1])
    h_next_min = tf.math.reduce_min(h_next[:-1])
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min

    # True if safe, False if dangerous
    is_safe_action = tf.math.less_equal(tf.constant(0, dtype=tf.float32), deriv)
    return is_safe_action, ph, deriv




def calc_deriv_prob(alpha, r, H=3):
    """
    Description:
        Calculate the derivative loss.

    Args:
        alpha: alpha CBF

    Returns:
        is_safe_action: bool
    """
    # placeholders
    s = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='state')
    s_next = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='next_state')
    a = tf.placeholder(tf.float32, [1, 2], name='action')
    ph = [s, s_next, a]

    # Get current timestep state of the agent
    s_agent = s[-1:, :]  # [1, 4]

    # Calculate next timestep state of the agent
    dsdt = tf.concat([s_agent[:, 2:], a], axis=1)
    s_agent_next = s_agent + dsdt * config.TIME_STEP  # [1, 4]

    # one more step for agent
    s_agent_2nd = s_agent_next[:, :2] + s_agent_next[:, 2:] * config.TIME_STEP * H

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:-1, :]  # [topk, 4]

    # one more step for obs
    s_obstacles_2nd = s_obstacles_next[:, :2] + s_obstacles_next[:, 2:] * config.TIME_STEP * H

    # The next state using combination
    s_next_real = tf.concat([s_obstacles_2nd, s_agent_2nd], axis=0)  # [topk + 1, 2]

    # Get h of current timestep
    x = tf.expand_dims(s[:, :2], 1) - tf.expand_dims(s[:, :2], 0)  # [topk + 1, topk + 1, 2]
    h = tf.reshape(network_cbf_vel(x=x, r=r)[0], [-1])

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 1) - tf.expand_dims(s_next_real, 0)  # [topk + 1, topk + 1, 2]
    h_next = tf.reshape(network_cbf_vel(x=x_next, r=r)[0], [-1])

    # Calculate h derivative, use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:-1])
    h_next_min = tf.math.reduce_min(h_next[:-1])
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min

    # True if safe, False if dangerous
    is_safe_action = tf.math.less_equal(tf.constant(0, dtype=tf.float32), deriv)
    return is_safe_action, ph, deriv




def calc_deriv_batch_real(alpha, batch_size):
    # placeholders
    s = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='state')
    s_next = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='state_next')
    ph = [s, s_next]

    # calc h & h_next
    h = CBF_NN(s, is_only_pos)  # [bs,]
    h_next = CBF_NN(s_next, is_only_pos)  # [bs,]

    # calc deriv
    deriv = h_next - h + config.TIME_STEP * alpha * h  # [bs,]

    # True if safe, False if dangerous
    is_safe_action = tf.math.less_equal(tf.zeros_like(deriv, dtype=tf.float32), deriv)  # [bs,]
    return is_safe_action, ph, deriv





def calc_deriv_batch(alpha, r, batch_size):
    """
    Description:
        Calculate the derivative loss.

    Args:
        alpha: alpha CBF

    Returns:
        is_safe_action: bool
    """
    # placeholders
    s = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='state')
    s_next = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='next_state')
    a = tf.placeholder(tf.float32, [batch_size, 1, 2], name='action')
    ph = [s, s_next, a]

    # stack states
    s = tf.stack([s for _ in range(batch_size)], axis=0)  # [bs, 13, 4]
    s_next = tf.stack([s_next for _ in range(batch_size)], axis=0)  # [bs, 13, 4]

    # Get current timestep state of the agent
    s_agent = s[:, -1:, :]  # [bs, 1, 4]

    # Calculate next timestep state of the agent
    dsdt = tf.concat([s_agent[:, :, 2:], a], axis=2)  # [bs, 1, 4]
    s_agent_next = s_agent + dsdt * config.TIME_STEP  # [bs, 1, 4]

    # one more step for agent
    s_agent_2nd = s_agent_next[:, :, :2] + s_agent_next[:, :, 2:] * config.TIME_STEP  # [bs, 1, 2]

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:, :-1, :]  # [bs, topk, 4]

    # one more step for obs
    s_obstacles_2nd = s_obstacles_next[:, :, :2] + s_obstacles_next[:, :, 2:] * config.TIME_STEP  # [bs, topk, 2]

    # The next state using combination
    s_next_real = tf.concat([s_obstacles_2nd, s_agent_2nd], axis=1)  # [bs, topk + 1, 2]

    # Get h of current timestep
    x = tf.expand_dims(s[:, :, :2], 2) - tf.expand_dims(s[:, :, :2], 1)  # [bs, topk + 1, topk + 1, 2]

    h = tf.compat.v1.map_fn(fn=lambda t: tf.reshape(network_cbf_vel(x=t, r=r)[0], [-1]),
                                 elems=x)  # [bs, topk + 1]

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 2) - tf.expand_dims(s_next_real, 1)  # [bs, topk + 1, topk + 1, 2]
    h_next = tf.compat.v1.map_fn(fn=lambda t: tf.reshape(network_cbf_vel(x=t, r=r)[0], [-1]),
                                 elems=x_next)  # [bs, topk + 1]

    # Calculate h derivative, use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:, :-1], axis=1)  # [bs,]
    h_next_min = tf.math.reduce_min(h_next[:, :-1], axis=1)  # [bs,]
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min  # [bs,]

    # True if safe, False if dangerous
    is_safe_action = tf.math.less_equal(tf.zeros_like(deriv, dtype=tf.float32), deriv)  # [bs,]
    return is_safe_action, ph, deriv





def calc_deriv_real_batch(alpha, r, batch_size):
    """
    Description:
        Calculate the derivative loss.

    Args:
        alpha: alpha CBF

    Returns:
        is_safe_action: bool
    """
    # placeholders
    s = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='state')
    s_next = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='next_state')
    a = tf.placeholder(tf.float32, [batch_size, 1, 2], name='action')
    ph = [s, s_next, a]

    # stack states
    s = tf.stack([s for _ in range(batch_size)], axis=0)  # [bs, 13, 4]
    s_next = tf.stack([s_next for _ in range(batch_size)], axis=0)  # [bs, 13, 4]

    # Get current timestep state of the agent
    s_agent = s[:, -1:, :]  # [bs, 1, 4]

    # Calculate next timestep state of the agent
    dsdt = tf.concat([s_agent[:, :, 2:], a], axis=2)  # [bs, 1, 4]
    s_agent_next = s_agent + dsdt * config.TIME_STEP  # [bs, 1, 4]

    # one more step for agent
    s_agent_2nd = s_agent_next[:, :, :2] + s_agent_next[:, :, 2:] * config.TIME_STEP  # [bs, 1, 2]

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:, :-1, :]  # [bs, topk, 4]

    # one more step for obs
    s_obstacles_2nd = s_obstacles_next[:, :, :2] + s_obstacles_next[:, :, 2:] * config.TIME_STEP  # [bs, topk, 2]

    # The next state using combination
    s_next_real = tf.concat([s_obstacles_2nd, s_agent_2nd], axis=1)  # [bs, topk + 1, 2]

    # Get h of current timestep
    x = tf.expand_dims(s[:, :, :2], 2) - tf.expand_dims(s[:, :, :2], 1)  # [bs, topk + 1, topk + 1, 2]
    h = tf.reshape(network_cbf_batch(x=x, r=r)[0], [batch_size, -1])  # [bs, topk + 1]


    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 2) - tf.expand_dims(s_next_real, 1)  # [bs, topk + 1, topk + 1, 2]
    h_next = tf.reshape(network_cbf_batch(x=x_next, r=r)[0], [batch_size, -1])  # [bs, topk + 1]

    # Calculate h derivative, use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:, :-1], axis=1)  # [bs,]
    h_next_min = tf.math.reduce_min(h_next[:, :-1], axis=1)  # [bs,]
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min  # [bs,]

    # True if safe, False if dangerous
    is_safe_action = tf.math.less_equal(tf.zeros_like(deriv, dtype=tf.float32), deriv)  # [bs,]
    return is_safe_action, ph, deriv





def calc_deriv_batch_gaussian(alpha, r, batch_size):
    """
    Description:
        Calculate the derivative loss. Load the GaussianConv1d NN.

    Args:
        alpha: alpha CBF

    Returns:
        is_safe_action: bool
    """
    # placeholders
    s = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='state')
    s_next = tf.placeholder(tf.float32, [config.TOP_K + 1, 4], name='next_state')
    a = tf.placeholder(tf.float32, [batch_size, 1, 2], name='action')
    ph = [s, s_next, a]

    # stack states
    s = tf.stack([s for _ in range(batch_size)], axis=0)  # [bs, 13, 4]
    s_next = tf.stack([s_next for _ in range(batch_size)], axis=0)  # [bs, 13, 4]

    # Get current timestep state of the agent
    s_agent = s[:, -1:, :]  # [bs, 1, 4]

    # Calculate next timestep state of the agent
    dsdt = tf.concat([s_agent[:, :, 2:], a], axis=2)  # [bs, 1, 4]
    s_agent_next = s_agent + dsdt * config.TIME_STEP  # [bs, 1, 4]

    # one more step for agent
    s_agent_2nd = s_agent_next[:, :, :2] + s_agent_next[:, :, 2:] * config.TIME_STEP  # [bs, 1, 2]

    # Get next timestep state of obstacles
    s_obstacles_next = s_next[:, :-1, :]  # [bs, topk, 4]

    # one more step for obs
    s_obstacles_2nd = s_obstacles_next[:, :, :2] + s_obstacles_next[:, :, 2:] * config.TIME_STEP  # [bs, topk, 2]

    # The next state using combination
    s_next_real = tf.concat([s_obstacles_2nd, s_agent_2nd], axis=1)  # [bs, topk + 1, 2]

    # Get h of current timestep
    x = tf.expand_dims(s[:, :, :2], 2) - tf.expand_dims(s[:, :, :2], 1)  # [bs, topk + 1, topk + 1, 2]

    h = tf.compat.v1.map_fn(fn=lambda t: tf.reshape(network_cbf_vel(x=t, r=r)[0], [-1]),
                                 elems=x)  # [bs, topk + 1]

    # Get h of next timestep state
    x_next = tf.expand_dims(s_next_real, 2) - tf.expand_dims(s_next_real, 1)  # [bs, topk + 1, topk + 1, 2]
    h_next = tf.compat.v1.map_fn(fn=lambda t: tf.reshape(network_cbf_vel(x=t, r=r)[0], [-1]),
                                 elems=x_next)  # [bs, topk + 1]

    # Calculate h derivative, use min(h(s)) as h(s)
    h_min = tf.math.reduce_min(h[:, :-1], axis=1)  # [bs,]
    h_next_min = tf.math.reduce_min(h_next[:, :-1], axis=1)  # [bs,]
    deriv = h_next_min - h_min + config.TIME_STEP * alpha * h_min  # [bs,]

    # True if safe, False if dangerous
    is_safe_action = tf.math.less_equal(tf.zeros_like(deriv, dtype=tf.float32), deriv)  # [bs,]
    return is_safe_action, ph, deriv