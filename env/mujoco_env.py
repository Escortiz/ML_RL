"""Wrapper class for MuJoCo environments"""

from collections import OrderedDict
import os
from os import path

from gym import error, spaces
from gym.utils import seeding
import numpy as np
import gym

_MUJOCO_BACKEND = None
_HAS_MUJOCO_PY = False
_HAS_MUJOCO = False
try:
    import mujoco_py
    _MUJOCO_BACKEND = "mujoco_py"
    _HAS_MUJOCO_PY = True
except Exception:
    try:
        import mujoco
        _MUJOCO_BACKEND = "mujoco"
        _HAS_MUJOCO = True
    except Exception:
        raise error.DependencyNotInstalled("You need to install mujoco_py or mujoco (https://mujoco.org/). In Colab prefer 'pip install mujoco' and set MUJOCO_GL=egl, or install mujoco_py locally if needed.")

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Interface for MuJoCo environments.
    """

    def __init__(self, frame_skip, render_mode=None):

        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.build_model()
        
        # Adaptar para ambas versiones de mujoco
        if _HAS_MUJOCO_PY:
            self.data = self.sim.data
            self.init_qpos = self.sim.data.qpos.ravel().copy()
            self.init_qvel = self.sim.data.qvel.ravel().copy()
        else:
            self.data = self.sim  # En la API moderna, sim es directamente el MjData
            self.init_qpos = self.sim.qpos.ravel().copy()
            self.init_qvel = self.sim.qvel.ravel().copy()

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, truncated, _info = self.step(action)
        assert not done and not truncated

        self._set_observation_space(observation)

        self.seed()

    def build_model(self):
        xml_path = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")
        # Backend: mujoco_py (legacy) or mujoco (modern)
        if _HAS_MUJOCO_PY:
            self.model = mujoco_py.load_model_from_path(xml_path)
            self.sim = mujoco_py.MjSim(self.model)
        else:
            # Use modern mujoco Python bindings. We implement the minimal set of
            # operations the environment needs (load model and create sim).
            # Note: rendering APIs differ between backends; advanced rendering
            # (MjViewer, offscreen) may be limited here.
            try:
                # try both common loader names
                if hasattr(mujoco, 'load_model_from_path'):
                    self.model = mujoco.load_model_from_path(xml_path)
                elif hasattr(mujoco, 'MjModel') and hasattr(mujoco.MjModel, 'from_xml_path'):
                    self.model = mujoco.MjModel.from_xml_path(xml_path)
                else:
                    # fallback to low-level load
                    self.model = mujoco.MjModel(xml_path)
                # create sim for modern mujoco (v3.x)
                self.sim = mujoco.MjData(self.model)
            except Exception as e:
                raise error.DependencyNotInstalled(f"Failed to initialize modern 'mujoco' backend: {e}")
        self.viewer = None
        self._viewers = {}

    def _set_action_space(self):
        if _HAS_MUJOCO_PY:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        else:
            # En mujoco moderno
            bounds = np.vstack((self.model.actuator_ctrlrange[:,0], self.model.actuator_ctrlrange[:,1])).T.astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        # Different backends expose different state APIs; prefer direct assignment
        # when using modern 'mujoco'. For mujoco_py we keep the older code.
        if _HAS_MUJOCO_PY:
            old_state = self.sim.get_state()
            new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                             old_state.act, old_state.udd_state)
            self.sim.set_state(new_state)
            self.sim.forward()
        else:
            # Modern mujoco: assign qpos/qvel directly and forward the sim
            try:
                # Para la versión moderna de mujoco
                self.sim.qpos[:] = qpos
                self.sim.qvel[:] = qvel
                mujoco.mj_forward(self.model, self.sim)
            except Exception as e:
                raise RuntimeError(f"Failed to set state on modern mujoco sim: {e}")

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        # stepping is similar in both backends
        try:
            if _HAS_MUJOCO_PY:
                self.sim.data.ctrl[:] = ctrl
                for _ in range(n_frames):
                    self.sim.step()
            else:
                # Para la versión moderna de mujoco
                self.sim.ctrl[:] = ctrl
                for _ in range(n_frames):
                    mujoco.mj_step(self.model, self.sim)
        except Exception as e:
            raise RuntimeError(f"Simulation step failed: {e}")

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if _HAS_MUJOCO_PY:
                if mode == 'human':
                    self.viewer = mujoco_py.MjViewer(self.sim)
                elif mode == 'rgb_array' or mode == 'depth_array':
                    self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            else:
                # Modern mujoco rendering differs; try to provide a helpful error
                # message. In many Colab/headless setups the modern 'mujoco'
                # package with MUJOCO_GL=egl can render offscreen, but adapting
                # the APIs here is non-trivial. If you need rendering in Colab,
                # prefer installing 'mujoco' + glfw and adjust this wrapper, or
                # install 'mujoco_py' locally where supported.
                raise RuntimeError("Rendering with the modern 'mujoco' backend is not fully supported by this wrapper. If you only need to run headless simulations, continue; otherwise install mujoco_py or extend this file to use the modern rendering APIs.")

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        if _HAS_MUJOCO_PY:
            return self.data.get_body_xpos(body_name)
        else:
            # En mujoco moderno, necesitamos obtener el ID del cuerpo primero
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            return self.data.xpos[body_id]

    def state_vector(self):
        if _HAS_MUJOCO_PY:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat
            ])
        else:
            # Mujoco moderno
            return np.concatenate([
                self.sim.qpos.flat,
                self.sim.qvel.flat
            ])
