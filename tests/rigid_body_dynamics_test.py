from numpy.testing import assert_almost_equal

import rigidbody.rigid_body_dynamics as rbd

import numpy as np

def test_simulate():
    t = np.linspace(0, 10, 100)

    coordinates_ = [0, 0, 0, 0, 0, 0]
    speeds_ = [1, 0, 0, 0, 0, 1]

    start = np.array(coordinates_ + speeds_)

    df = rbd.simulate(t=t, force_torque=[0.3, 0.6, 0.5, 0.2, 0.1, 0.1], I_xx=1, I_yy=1, I_zz=1, mass=1,
                  initial_speeds=[1, 0.01, 0.01, 0.01, 0.01, 0.2], initial_coordinates=[0, 0, 0, 0, 0, 0.0])

def test_ball_drop():
    """
    Simulate a ball drop and check the result

    :param point:
    :return:
    """

    mass = 10
    g = 9.81
    t = np.linspace(0, 10, 100)

    f = mass*g
    force_torque = [0,0,f,0,0,0]
    df = rbd.simulate(t=t, force_torque=force_torque, I_xx=1, I_yy=1, I_zz=1, mass=mass)

    position = df.iloc[-1][['x0','y0','z0','phi','theta','psi']]

    s = g*t[-1]**2/2
    exprected_position = [0,0,s,0,0,0]

    assert_almost_equal(position, exprected_position)
