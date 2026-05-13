import numpy as np

_L, _R, _U, _D = ord('a'), ord('d'), ord('w'), ord('s')

# Per-environment keyboard → action-vector mapping.
# No key held → zero action (let physics coast).
ENV_KEY_MAPS = {
    'Pendulum-v1': {
        _L: np.array([-2.0]),
        _R: np.array([ 2.0]),
    },
    'MountainCarContinuous-v0': {
        _L: np.array([-1.0]),
        _R: np.array([ 1.0]),
    },
    'InvertedPendulum-v4': {
        _L: np.array([-1.0]),
        _R: np.array([ 1.0]),
    },
    'InvertedDoublePendulum-v4': {
        _L: np.array([-1.0]),
        _R: np.array([ 1.0]),
    },
    'Reacher-v4': {
        _L: np.array([-1.0,  0.0]),
        _R: np.array([ 1.0,  0.0]),
        _U: np.array([ 0.0,  1.0]),
        _D: np.array([ 0.0, -1.0]),
    },
    'Swimmer-v4': {
        _L: np.array([-1.0,  0.0]),
        _R: np.array([ 1.0,  0.0]),
        _U: np.array([ 0.0,  1.0]),
        _D: np.array([ 0.0, -1.0]),
    },
    # dims: thigh, leg, foot
    'Hopper-v4': {
        ord('w'): np.array([ 1.0,  1.0,  1.0]),
        ord('s'): np.array([-1.0, -1.0, -1.0]),
        _R:       np.array([ 1.0,  0.0,  0.0]),
        _L:       np.array([-1.0,  0.0,  0.0]),
    },
    # dims: back-thigh, back-shin, back-foot, front-thigh, front-shin, front-foot
    'HalfCheetah-v4': {
        ord('w'): np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
        ord('s'): np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        _R:       np.array([ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0]),
        _L:       np.array([ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
    },
    # dims: right-thigh, right-leg, right-foot, left-thigh, left-leg, left-foot
    'Walker2d-v4': {
        ord('w'): np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
        ord('s'): np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        _R:       np.array([ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
        _L:       np.array([ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0]),
    },
    # dims: 4 hip joints + 4 ankle joints
    'Ant-v4': {
        ord('w'): np.array([ 1.0,  0.0, -1.0,  0.0,  1.0,  1.0,  1.0,  1.0]),
        ord('s'): np.array([-1.0,  0.0,  1.0,  0.0, -1.0, -1.0, -1.0, -1.0]),
        ord('a'): np.array([ 0.0,  1.0,  0.0, -1.0,  1.0,  1.0,  1.0,  1.0]),
        ord('d'): np.array([ 0.0, -1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
    },
    'Humanoid-v4': {
        ord('w'): np.full(17,  0.4),
        ord('s'): np.full(17, -0.4),
        ord(' '): np.zeros(17),
    },
    'HumanoidStandup-v4': {
        ord('w'): np.full(17,  0.4),
        ord('s'): np.full(17, -0.4),
        ord(' '): np.zeros(17),
    },
}

ENV_HELP = {
    'Pendulum-v1':               'a/d: torque',
    'MountainCarContinuous-v0':  'a/d: push',
    'InvertedPendulum-v4':       'a/d: force',
    'InvertedDoublePendulum-v4': 'a/d: force',
    'Reacher-v4':                'a/d: joint0  w/s: joint1',
    'Swimmer-v4':                'a/d: joint0  w/s: joint1',
    'Hopper-v4':                 'w/s: all joints  a/d: thigh',
    'HalfCheetah-v4':            'w/s: all joints  a/d: back/front thigh',
    'Walker2d-v4':               'w/s: all joints  a/d: right/left hip',
    'Ant-v4':                    'w/a/s/d: directional movement',
    'Humanoid-v4':               'w/s: all joints  space: zero',
    'HumanoidStandup-v4':        'w/s: all joints  space: zero',
}
