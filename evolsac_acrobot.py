import os

import numpy as np
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array
from double_pendulum.simulation.simulation import Simulator

# from double_pendulum.controller.evolsac.evolsac_controller import EvolSACController
from evolsaccontroller import EvolSACController
from stable_baselines3 import SAC

design = "design_C.1"


friction_compensation = True
robot = "acrobot"
torque_limit = [0.0, 3.0]

# trajectory
dt = 0.005
t_final = 10.0
N = int(t_final / dt)
T_des = np.linspace(0, t_final, N + 1)
u1 = np.zeros(N + 1)
u2 = np.zeros(N + 1)
U_des = np.array([u1, u2]).T

# controller
# learning environment parameters
state_representation = 2

model_par_path = "model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
plant = SymbolicDoublePendulum(model_pars=mpar)
print("Loading model parameters...")
simulator = Simulator(plant=plant)

dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator="runge_kutta",
    robot=robot,
    state_representation=state_representation,
    max_velocity=50.0,
    torque_limit=torque_limit,
)

dt = 1 / 500  # 500 Hz
control_frequency = 1 / 100  # 100 Hz controller frequency

ctrl_rate = int(control_frequency / dt)

# dovrebbe essere quello corretto per la performance leaderboard (senza attrito)
model_path = "policies/model_1.0/acrobot/evolsac/model.zip"

sac_model = SAC.load(model_path)
controller = EvolSACController(
    model=sac_model,
    dynamics_func=dynamics_func,
    window_size=0,
    include_time=False,
    ctrl_rate=5,
    wait_steps=0,
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
# run_experiment(
#     controller=controller,
#     dt=dt,
#     t_final=t_final,
#     can_port="can0",
#     motor_ids=[3, 1],
#     tau_limit=torque_limit,
#     save_dir=os.path.join("data", design, f"{robot}/evolsac_{"FC" if friction_compensation else ""}"),
#     record_video=True,
#     safety_velocity_limit=30.0,
#     perturbation_array=perturbation_array,
# )
