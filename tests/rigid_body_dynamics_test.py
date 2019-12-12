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

def test_circle():
    """
    Simulate a point mass traveling in a steady circle
    :return:
    """

    radius = 10  # Radius of rotation [m]
    w = 0.1  # Angle velocity [rad/s]
    V = radius * w  # Speed of point [m/s]
    t = np.linspace(0, 2 * np.pi / w, 1000)

    mass = 1

    expected_acceleration = -radius * w ** 2
    expected_force = mass * expected_acceleration

    df = rbd.simulate(t=t, force_torque=[0, -expected_force, 0, 0, 0, 0], I_xx=1, I_yy=1, I_zz=1, mass=mass,
                  initial_speeds=[V, 0, 0, 0, 0, w], initial_coordinates=[0, -radius, 0, 0, 0, 0])

    R = np.sqrt(df['x0'] ** 2 + df['y0'] ** 2)
    expected_R = np.ones(len(df))*radius

    assert_almost_equal(R, expected_R, decimal=6)  # Make sure that the simulation has a steady turning diameter.

def test_angular_acceleration():

    t = np.linspace(0, 10, 100)
    mass = 2

    I_zz = 10

    torque_yaw = 1
    df = rbd.simulate(t=t, force_torque=[0, 0, 0, 0, 0, torque_yaw], I_xx=1, I_yy=1, I_zz=I_zz, mass=mass)

    w1d = torque_yaw/I_zz
    w = w1d*t

    assert_almost_equal(df['psi1d'], w)
