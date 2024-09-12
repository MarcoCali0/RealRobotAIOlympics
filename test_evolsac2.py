import argparse
import os

import gymnasium as gym
import numpy as np
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from stable_baselines3 import SAC

from evolsaccontroller import EvolSACController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run double pendulum control experiment."
        + "\nExample: python script.py pendubot --model_selection 0 --lqr True --friction_compensation False",
    )

    parser.add_argument(
        "robot",
        type=str,
        choices=["acrobot", "pendubot"],
        help="Type of robot: acrobot or pendubot",
    )
    parser.add_argument(
        "--friction_compensation",
        type=bool,
        choices=[True, False],
        help="Enable friction compensation.",
    )
    parser.add_argument(
        "--model_selection",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
        help="Select model (0-5).",
    )
    parser.add_argument(
        "--lqr",
        type=bool,
        choices=[True, False],
        help="Enable LQR controller for stabilisation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot = args.robot
    friction_compensation = args.friction_compensation
    model_selection = args.model_selection
    LQR_enabled = args.lqr

    # New models have all max_torque = 3.0, max_velocity = 50.0, window_size = 0
    max_torque = 3.0
    max_velocity = 50.0
    window_size = 0
    include_time = False
    torque_limit = [0.0, max_torque] if robot == "acrobot" else [max_torque, 0.0]

    # Time parameters
    dt = 1 / 500
    t_final = 10.0
    N = int(t_final / dt)
    T_des = np.linspace(0, t_final, N + 1)
    U_des = np.zeros((N + 1, 2))

    # Load model parameters and setup plant
    model_par_path = "model_parameters.yml"
    mpar = model_parameters(filepath=model_par_path)
    plant = SymbolicDoublePendulum(model_pars=mpar)

    # Simulation setup
    simulator = Simulator(plant=plant)
    integrator = "runge_kutta"
    state_representation = 2

    dynamics_func = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot=robot,
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=torque_limit,
    )

    # # Controller frequency
    control_frequency = 1 / 100
    ctrl_rate = int(control_frequency / dt)

    # Model path based on model selection to define based on the found models
    model_paths = {
        0: f"new_models/{robot}_no_friction",
        1: f"new_models/{robot}_noisy",
    }
    model_path = model_paths.get(model_selection)

    # Load SAC model
    obs_space = gym.spaces.Box(np.array([-1.0] * 4), np.array([1.0] * 4))
    act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
    sac_model = SAC.load(
        model_path,
        custom_objects={"observation_space": obs_space, "action_space": act_space},
    )

    # LQR controller parameters
    lqr_path = f"lqr_{robot}"
    lqr_pars = np.loadtxt(os.path.join(lqr_path, "controller_par.csv"))
    Q_lqr = np.diag(lqr_pars[:4])
    R_lqr = 8.0 * np.diag([lqr_pars[4], lqr_pars[4]])

    S = np.loadtxt(os.path.join(lqr_path, "Smatrix"))
    rho = np.loadtxt(os.path.join(lqr_path, "rho"))

    # Define switching conditions
    def condition1(t, x):
        theta1, theta2, _, _ = x
        y_ee = dynamics_func.simulator.plant.forward_kinematics([theta1, theta2])[1][1]
        if y_ee <= 0.2:
            print(f"Switching to SAC controller at y_ee = {y_ee}")
            return True

        return False

    # default condition set in last year scripts
    def condition2(t, x):
        goal = [np.pi, 0.0, 0.0, 0.0]
        delta = wrap_angles_diff(np.subtract(x, goal))
        if np.einsum("i,ij,j", delta, S, delta) < rho:
            print(f"Switch to LQR at time = {t}")
            return True and LQR_enabled
        return False

    # SAC controller
    controller1 = EvolSACController(
        model=sac_model,
        dynamics_func=dynamics_func,
        window_size=window_size,
        include_time=include_time,
        ctrl_rate=ctrl_rate,
        wait_steps=0,
    )

    # LQR controller
    controller2 = LQRController(model_pars=mpar)
    controller2.set_goal([np.pi, 0.0, 0.0, 0.0])
    controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
    controller2.set_parameters(failure_value=0.0, cost_to_go_cut=100000)

    # Combined controller
    controller = CombinedController(
        controller1=controller1,
        controller2=controller2,
        condition1=condition1,
        condition2=condition2,
        compute_both=False,
        verbose=True,
    )

    # Friction compensation
    if friction_compensation:
        controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)

    # Low-pass filter
    filter = lowpass_filter(
        alpha=[1.0, 1.0, 0.2, 0.2], x0=[0.0, 0.0, 0.0, 0.0], filt_velocity_cut=0.1
    )
    controller.set_filter(filter)
    controller.init()

    # Perturbations
    perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
        t_final, dt, 2, 1.0, [0.05, 0.1], [0.4, 0.6]
    )

    print(
        f"""
    Running model: {model_selection}
    Robot: {robot}
    Friction compensation: {'Enabled' if friction_compensation else 'Disabled'}
    LQR: {'Enabled' if LQR_enabled else 'Disabled'}
    Max torque: {max_torque} Nm
    Loop frequency: {dt} s
    Controller frequency: {control_frequency} s
    """
    )

    # Run experiment
    run_experiment(
        controller=controller,
        dt=dt,
        t_final=t_final,
        can_port="can0",
        motor_ids=[3, 1],
        motor_directions=[1.0, -1.0],
        tau_limit=[6.0, 0.5] if robot == "pendubot" else [0.5, 6.0],
        save_dir=os.path.join(
            "data",
            f"{robot}/evolsac_{max_torque}Nm_{model_path}{'_FC' if friction_compensation else ''}",
        ),
        record_video=False,
        safety_velocity_limit=30.0,
        perturbation_array=perturbation_array,
    )


main()
