#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:22:31 2019

@author: rajan
"""

import numpy as np;
import scipy;
from scipy.integrate import ode;
import matplotlib.pyplot as plt;
import control; # control systems library
import slycot;
from control import matlab;

"""

===== MODEL SETUP =====

- geometric parameters of the robot
- mass/inertia properties

"""

m1 = .200; # kg
l1 = .200; # m
w1 = .030; # m
J1 = 1/12*m1*(w1**2 + l1**2); # kg-m^2

m2 = .300; # kg
r2 = .050; # m
J2 = 1/2*m2*r2**2; # kg-m^2

g = 9.81; # m/s^2

# not modeling friction for now...


"""
===== STATE SPACE LINEARIZATION =====

- this simulation is for upright control only (linearized region)

- state space:  q1 = link angle from pure upright, according to RHR
                q2 = flywheel angle from arbitrary 0
                q1_dot = link angle angular velocity
                q2_dot = flywheel angular velocity

- command space: u = motor torque on flywheel

- state space form: X_dot = A*x + B*u

- feedback: Y = C*x + D*u

- how do we estimate q?
    q1 -> IMU gives direction of g vector
    q2 -> we don't need to! 2nd col in A matrix is 0.
    q1_dot -> IMU gives "yaw rate"
    q2_dot -> quad armature encoder on DC motor gives shaft angular velocity

- so, we have full state feedback, in the linear domain near the unstable
  equilibrium.


"""


# fully linear system

# some coefficients to make our life easier
a = m1*g*l1/2 + m2*g*l1;
b = m1*l1**2/4 + J1 + m2*l1**2 + J2;

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [a/(b-J2), 0, 0, 0],
              [a/(J2-b), 0, 0, 0]]);

B = np.array([[0],
              [0],
              [-1/(b-J2)],
              [-b/(J2*(J2-b))]]);


C = np.eye(4);
D = np.zeros((1, 1));

#print(A);
#print(B);
#print(C);
#print(D);


controllability_rank = np.linalg.matrix_rank(control.ctrb(A, B)); # should be 4
print(controllability_rank);



Q = np.array([[10, 0, 0, 0],
              [0, 10, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 10]]); # state cost matrix


# comically large cost...
R = 10000000000*np.eye(1); # control input cost matrix


K, S, E = control.lqr(A, B, Q, R);

print(K);

# u = -K*(q_current - q_desired)

# now K is our gain matrix

# simulation upright stabilization

q_final = np.array([[0],
                    [0],
                    [0],
                    [0]]);

q_initial = np.array([[30*np.pi/180],
                      [0],
                      [0],
                      [0]]);

"""
===== MANUAL INTEGRATION =====
- Currently using Eulerian integration
- On deck.. Mr. Runge-Kutta for better accuracy/computing time ratio.
"""


dt = .001; #sec
time = 5; #sec
n = int(time/dt); # number of samples

tspan = np.linspace(0, time, num=n);
state_evolution = np.zeros((4, n)); # time-series of the state space
state_evolution[:, [0]] = q_initial;
dq_evolution = np.zeros((4, n));

control_evolution = np.zeros((1, n)); # time series of the requested torque


for i in range(1, n//2):

    q_old = state_evolution[:, [i-1]];


    u = np.matmul(-1*K, (q_old-q_final));

    dq = np.matmul(A, q_old) + np.matmul(B,u);
    state_evolution[:, [i]] = q_old + dq*dt;
    control_evolution[:, i] = u;
    dq_evolution[:, [i]] = dq;
    #x_loc = l1*np.sin(state_evolution[0, i]);
    #y_loc = l1*np.cos(state_evolution[1, i]);

   # print(i);

#

#print(control_evolution[0:5]);
#print(state_evolution[:, 0:5]);
#print(dq_evolution[:, 0:5]);


state_evolution[:, [n//2]] = -q_initial;
control_evolution[:, [n//2]] = 0;
for i in range(n//2 + 1, n):
    q_old = state_evolution[:, [i-1]];


    u = np.matmul(-1*K, (q_old-q_final));

    dq = np.matmul(A, q_old) + np.matmul(B,u);
    state_evolution[:, [i]] = q_old + dq*dt;
    control_evolution[:, i] = u;



plt.subplot(2, 1, 1);

plt.plot(tspan, state_evolution[0, :]*180/np.pi);
plt.plot(tspan, state_evolution[2, :]*180/np.pi);
plt.xlabel("Time [s]");
plt.ylabel("Pendulum Angle [deg]");
plt.axhline(q_final[0], linestyle='dashed');
plt.axhline(q_final[2], linestyle='dashed');
plt.grid(1);
plt.title("State Space Evolution, Step Response");
plt.legend(["Pendulum Angle", "Flywheel Angle"]);
plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')



plt.subplot(2, 1, 2);
plt.plot(tspan, control_evolution[0, :]);
plt.grid(1);
plt.xlabel("Time [s]");
plt.ylabel("Torque Input [N-m]");
plt.title("State Space Evolution, Step Response");

plt.tight_layout();
plt.show();




#plt.figure(1);
#plt.plot(pose_evolution[0, :], pose_evolution[1, :]);
#plt.xlabel("x-coordinate [m]");
#plt.ylabel("y-coordinate [m]");
#plt.title("Robot Path, Step Response");
#plt.axis("equal");
#
#plt.figure(2);
#plt.plot(tspan, torque_evolution[0, :]);
#plt.plot(tspan, torque_evolution[1, :]);
#plt.xlabel("Time [s]");
#plt.ylabel("Torque [N-m]");
#plt.legend(["Left Wheel Torque", "Right Wheel Torque"]);
#plt.title("Controller Torque Evolution, Step Response");


"""
tspan = np.linspace(0, 10, num=200); # simulate 10 seconds at .1 sec intervals
x0 = np.array([0, 0]).T; # start off with no initial velocities


ss_ol = control.matlab.ss(A, B, C, D); # open loop state space
A_matrix_cl = A - np.matmul(B, K);
ss_cl = control.matlab.ss(A_matrix_cl, B, C, D); # closed loop state space, np.matmul(B, K) for B

reference = np.ones((2, tspan.shape[0]));
reference[0, :] *= lin_vel_reference;
reference[1, :] *= ang_vel_reference;
"""


"""
yout, tout, xout = control.matlab.lsim(ss_cl, reference.T, tspan, X0=0.0)

plt.figure(0);
plt.plot(tout, reference[0, :]);
plt.plot(tout, reference[1, :]);
plt.plot(tout, yout[:, 0]);
plt.plot(tout, yout[:, 1]);


plt.figure(1);
plt.plot(tout, xout[:, 0]);
plt.plot(tout, xout[:, 1]);
"""


"""

# controller command is -K*(x-xf);


def dx(u):
    dx = np.matmul(B, u);
    return dx;


solver = ode(dx);
solver.set_integrator('dop853');
solver.set_initial_value(x0, 0.0);
sol = np.empty((100, 2));
sol[0] = x0;

k = 1;
while solver.successful() and solver.tspan < 10:
    solver.integrate(tspan[k]);
    sol[k] = solver.y;
    k += 1;
"""

"""

MATLAB implementation:

function dy = two_wheeled_robot(y, mass, rot_inertia,  F_rr, track, u)
dy(1, 1) = (u(1) + u(2) - F_rr)/mass;
dy(2, 1) = -y(1)*sin(y(3));
dy(3, 1) = y(4);
dy(4, 1) = (u(2) - u(1))*track/2/rot_inertia;
end

"""
