import os
import sys

import gymnasium as gym
import numpy as np
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array
from double_pendulum.simulation.simulation import Simulator
from stable_baselines3 import SAC

# from double_pendulum.controller.evolsac.evolsac_controller import EvolSACController
from evolsaccontroller import EvolSACController

robot = str(sys.argv[1])
friction_compensation = True if str(sys.argv[2]) == "FC" else False
model_selection = int(sys.argv[3])

max_torque = 5.0
torque_limit = [0.0, max_torque] if robot == "acrobot" else [max_torque, 0.0]

# trajectory
dt = 0.005
t_final = 10.0
N = int(t_final / dt)
T_des = np.linspace(0, t_final, N + 1)
u1 = np.zeros(N + 1)
u2 = np.zeros(N + 1)
U_des = np.array([u1, u2]).T

state_representation = 2
integrator = "runge_kutta"

model_par_path = "model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
plant = SymbolicDoublePendulum(model_pars=mpar)

print("Loading model parameters...")
simulator = Simulator(plant=plant)

max_velocity = 50.0  # usually 50.0
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

# dt = 1 / 500  # 500 Hz
control_frequency = 1 / 100  # 100 Hz controller frequency
ctrl_rate = int(control_frequency / dt)

wait_steps = 0

if model_selection == 0:
    model_path = f"models/{robot}_no_friction"
elif model_selection == 1:
    model_path = f"models/{robot}_noisy"
elif model_selection == 2:
    model_path = f"models/{robot}_noise_trained"
elif model_selection == 3:
    # trained with max_velocity = 20, max_torque = 3
    model_path = f"models/{robot}_model_to_try"


if model_selection not in [2, 3]:
    sac_model = SAC.load(model_path)
else:
    # this controller requires redefining obs and act spaces
    if model_selection in [3]:
        obs_space = gym.spaces.Box(np.array([-1.0] * 50), np.array([1.0] * 50))
    else:
        obs_space = gym.spaces.Box(np.array([-1.0] * 4), np.array([1.0] * 4))
    act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
    sac_model = SAC.load(
        model_path,
        custom_objects={"observation_space": obs_space, "action_space": act_space},
    )


controller = EvolSACController(
    model=sac_model,
    dynamics_func=dynamics_func,
    window_size=0,
    include_time=False,
    ctrl_rate=ctrl_rate,
    wait_steps=wait_steps,
)


# if policy trained without friction, keep whole friction vector
if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)


filter = lowpass_filter(
    alpha=[1.0, 1.0, 0.2, 0.2], x0=[0.0, 0.0, 0.0, 0.0], filt_velocity_cut=0.1
)

controller.set_filter(filter)
controller.init()


perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
    t_final, dt, 2, 1.0, [0.05, 0.1], [0.4, 0.6]
)

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[3, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=torque_limit,
    save_dir=os.path.join(
        "data",
        f"{robot}/evolsac_{max_torque}Nm_{model_path}{'_FC' if friction_compensation else ''}",
    ),
    record_video=False,
    safety_velocity_limit=30.0,
    #    perturbation_array=perturbation_array,
)
