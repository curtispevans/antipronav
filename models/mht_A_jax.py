import numpy as np
from jax import numpy as jnp
from jax import lax, grad, jacfwd
import jax
from functools import partial
from models.ekf_modified_polar_coordinates_knownA_jax import kalman_update as ekf_mpc_update
from models.nearly_constant_accel_kf_jax import kalman_update as kf_nca_update
from jax import config
config.update('jax_enable_x64', True)


def get_position_of_intruder(state, mav):
    distance = 1/state[3]  # inverse distance to distance
    bearing = state[2]  # bearing in radians
    own_pose = mav[0:2]  # own position
    own_heading = mav[2]  # own heading in radians
    los = jnp.array([jnp.cos(bearing + own_heading), jnp.sin(bearing + own_heading)])
    intruder_pos = own_pose + distance * los
    return intruder_pos

def get_mahalanobis_distance_intruder_state(state, sigma, measurement, R):
    C = jnp.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    hx = C @ state
    innovation = measurement - hx
    
    S = C @ sigma @ C.T + R
    
    D2 = innovation.T @ jnp.linalg.inv(S) @ innovation
    # print(D2)
    return D2


def update_all_filters(mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, bearing_pixel_measurement, Ts, mav, u, A):
    '''
    Update both MPC and NCA filters with the same measurement.
    mu_mpc, sigma_mpc: state and covariance for MPC
    mu_nca, sigma_nca: state and covariance for NCA
    Q_mpc, R_mpc: process and measurement noise for MPC
    Q_nca, R_nca: process and measurement noise for NCA
    bearing_pixel_measurement: measurement to update with
    Ts: time step
    mav: own MAV state
    u: control input
    A: constant used in the MPC measurement model
    '''
    
    # Update MPC filter
    mu_mpc, sigma_mpc = ekf_mpc_update(mu_mpc, sigma_mpc, mav, u, bearing_pixel_measurement, Q_mpc, R_mpc, Ts, A)
    

    # Get the position of the intruder using the MPC state
    intruder_pos = get_position_of_intruder(mu_mpc, mav)
    
    # Update NCA filter
    mu_nca, sigma_nca = kf_nca_update(mu_nca, sigma_nca, intruder_pos, Q_nca, R_nca, Ts)

    # compute mahalanobis distance
    D2 = get_mahalanobis_distance_intruder_state(mu_nca, sigma_nca, intruder_pos, R_nca)

    return mu_mpc, sigma_mpc, mu_nca, sigma_nca, D2

def wrapper_update_all_filters(mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, measurement, Ts, mav, u, A):
    '''
    Wrapper function to update all filters with the same measurement.
    This is used to compute the Jacobian of the update function.
    '''
    return update_all_filters(mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, measurement, Ts, mav, u, A)[-1]


@partial(jax.jit, static_argnames=('num_steps',))
def total_loss(A, initial_states, measurements, mav_states, controls, Ts,
               Q_mpc, R_mpc, Q_nca, R_nca, num_steps):
      '''
      Computes total Mahalanobis distance loss over multiple timesteps.
      '''
      mu_mpc, sigma_mpc, mu_nca, sigma_nca = initial_states

      def step(carry, t):
          mu_mpc, sigma_mpc, mu_nca, sigma_nca, total_D2 = carry

          # update filters
          results = update_all_filters(
              mu_mpc, sigma_mpc, mu_nca, sigma_nca,
              Q_mpc, R_mpc, Q_nca, R_nca,
              measurements[t], Ts, mav_states[t], controls[t], A
          )
          mu_mpc, sigma_mpc, mu_nca, sigma_nca, D2 = results

          # accumulate total loss
          total_D2 += D2

          return (mu_mpc, sigma_mpc, mu_nca, sigma_nca, total_D2), None
      
      # Initalize carry
      carry_init = (mu_mpc, sigma_mpc, mu_nca, sigma_nca, 0.0)

      # Run through all time steps
      (_, _, _, _, total_D2), _ = lax.scan(step, carry_init, jnp.arange(num_steps))

      return total_D2

def optimize_A(initial_states, measurements, mav_states, controls, Ts,
               Q_mpc, R_mpc, Q_nca, R_nca, num_steps, A_init=20.0,
               learning_rate=0.01, num_iters=100):
      '''
      Optimize parameter A using gradient descent.
      '''
      # create loss function with fixed parameters
      loss_fn = lambda A: total_loss(
         A, initial_states, measurements, mav_states, controls, Ts,
         Q_mpc, R_mpc, Q_nca, R_nca, num_steps
    )
      # JIT compile the loss and gradient functions
      loss_fn_jit = jax.jit(loss_fn)
      grad_fn = jax.jit(grad(loss_fn))

      # Gradient descent
      A_opt = A_init
      history = []

      for i in range(num_iters):
           # Compute loss and gradient
           loss_val = loss_fn_jit(A_opt)
           grad_val = grad_fn(A_opt)

           # Update parameter
           A_opt -= learning_rate * grad_val

           # store history
           history.append((i, A_opt, loss_val, grad_val))

           # print progress
           if i % 10 == 0:
                print(f"Iter {i}: A={A_opt:.4f}, Loss={loss_val:.4f}, Grad={grad_val:.4f}")

      return A_opt, history
