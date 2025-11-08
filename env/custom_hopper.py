"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv, _HAS_MUJOCO_PY, _HAS_MUJOCO

# Import mujoco if available (modern backend)
if _HAS_MUJOCO:
    import mujoco


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, render_mode=None):
        MujocoEnv.__init__(self, frame_skip=4, render_mode=render_mode)
        utils.EzPickle.__init__(self)

        if _HAS_MUJOCO_PY:
            self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
            if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
                self.sim.model.body_mass[1] *= 0.7
        else:
            # Mujoco moderno
            self.original_masses = np.copy(self.model.body_mass[1:])    # Default link masses
            if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
                self.model.body_mass[1] *= 0.7

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        
        #
        # TASK 6: implement domain randomization. Remember to sample new dynamics parameter
        #         at the start of each training episode.
        
        raise NotImplementedError()

        return


    def get_parameters(self):
        """Get value of mass for each link"""
        if _HAS_MUJOCO_PY:
            masses = np.array(self.sim.model.body_mass[1:])
        else:
            # Mujoco moderno
            masses = np.array(self.model.body_mass[1:])
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        if _HAS_MUJOCO_PY:
            self.sim.model.body_mass[1:] = task
        else:
            # Mujoco moderno
            self.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        if _HAS_MUJOCO_PY:
            posbefore = self.sim.data.qpos[0]
            self.do_simulation(a, self.frame_skip)
            posafter, height, ang = self.sim.data.qpos[0:3]
        else:
            # Mujoco moderno
            posbefore = self.sim.qpos[0]
            self.do_simulation(a, self.frame_skip)
            posafter, height, ang = self.sim.qpos[0:3]
            
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        # Devolver 4-tupla compatible con OpenAI Gym (obs, reward, done, info)
        # Para mantener compatibilidad con gym 0.21 y wrappers como TimeLimit
        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        if _HAS_MUJOCO_PY:
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat
            ])
        else:
            # Mujoco moderno
            return np.concatenate([
                self.sim.qpos.flat[1:],
                self.sim.qvel.flat
            ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        if _HAS_MUJOCO_PY:
            self.viewer.cam.trackbodyid = 2
            self.viewer.cam.distance = self.model.stat.extent * 0.75
            self.viewer.cam.lookat[2] = 1.15
            self.viewer.cam.elevation = -20
        else:
            # Modern mujoco Renderer doesn't have a 'cam' attribute
            # Camera setup is handled differently in the render() method
            pass


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        if _HAS_MUJOCO_PY:
            mjstate = deepcopy(self.get_mujoco_state())
            mjstate.qpos[0] = 0.
            mjstate.qpos[1:] = state[:5]
            mjstate.qvel[:] = state[5:]
            self.set_sim_state(mjstate)
        else:
            # Modern mujoco: directly set qpos and qvel
            self.sim.qpos[0] = 0.
            self.sim.qpos[1:] = state[:5]
            self.sim.qvel[:] = state[5:]
            mujoco.mj_forward(self.model, self.sim)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        if _HAS_MUJOCO_PY:
            return self.sim.set_state(mjstate)
        else:
            # Modern mujoco: state is set directly via qpos/qvel
            raise NotImplementedError("set_sim_state not supported with modern mujoco backend. Use set_state() instead.")


    def get_mujoco_state(self):
        """Returns current mjstate"""
        if _HAS_MUJOCO_PY:
            return self.sim.get_state()
        else:
            # Modern mujoco doesn't have get_state(), return a simple state dict
            class MjState:
                def __init__(self, qpos, qvel):
                    self.qpos = qpos.copy()
                    self.qvel = qvel.copy()
            return MjState(self.sim.qpos, self.sim.qvel)



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

