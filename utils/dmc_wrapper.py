# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple
import numpy as np
from dm_env import StepType, specs

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    state: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    # def __getitem__(self, attr):
    #     return getattr(self, attr)


class ExtendedTimeStepWrapper():
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self._env.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)


class ActionDTypeWrapper():
    def __init__(self, env, dtype, num_env):
        self._env = env
        wrapped_action_spec = self._env.action_space.shape
        self._action_spec = specs.BoundedArray(shape=np.array([num_env, wrapped_action_spec[0]]),
                                               dtype=dtype,
                                               minimum=self._env.action_space.low,
                                               maximum=self._env.action_space.high,
                                               name='action')

    def step(self, action):
        # action = action.astype(self.action_spec().dtype)
        action = action.float()
        return self._env.step(action)

    def observation_spec(self):
        return specs.BoundedArray(shape=np.array(self._env.observation_space.shape),
                                  dtype=np.float32,
                                  minimum=-np.inf,
                                  maximum=np.inf,
                                  name='observation')

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper():
    def __init__(self, env, num_repeats, num_envs):
        self._env = env
        self._num_repeats = num_repeats
        self.num_envs = num_envs

    def step(self, action):
        # TODO change the reward shape to the multi-env setting, and deal with the setting for self._num_repeat > 1 (i.e., dones)
        # TODO for self._num_repeat > 1, the info['success'] should be the sum of the success
        reward = 0.0
        discount = 1.0

        assert self._num_repeats > 0

        for i in range(self._num_repeats):
            next_obs, rews, dones, infos = self._env.step(action)
            next_state = self._env.get_state()
            reward += (rews) * discount
            # reward += (rews or 0.0) * discount
            # if time_step.last():
            #     break
        time_step = self._augment_time_step(next_obs.cpu().data.numpy(),
                                            next_state.cpu().data.numpy(),
                                            reward.cpu().data.numpy(),
                                            discount,
                                            dones.cpu().data.numpy(),
                                            action.cpu().data.numpy())

        return time_step, infos

    def _augment_time_step(self, obses, states, rews=0.0, discounts=1.0, dones=False, action=None):
        if action is None:
            action_spec = self._env.action_spec()
            action = np.zeros((self.num_envs, action_spec.shape[1]), dtype=action_spec.dtype)
            dones = np.zeros((self.num_envs, ), dtype=np.int8)
            rews = np.zeros((self.num_envs, ), dtype=np.float32)
        return ExtendedTimeStep(observation=obses,
                                step_type=dones,
                                action=action,
                                reward=rews,
                                discount=discounts or 1.0,
                                state=states)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    # TODO modefying to mutil env settings
    def reset(self):
        obses = self._env.reset()
        state = self._env.get_state()
        return self._augment_time_step(obses.cpu().data.numpy(), state.cpu().data.numpy())

    def __getattr__(self, name):
        return getattr(self._env, name)




class FrameStackWrapper():
    def __init__(self, env, num_frames, num_envs, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key
        self.num_envs = num_envs
        self._state_frames = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()

        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        # if len(pixels_shape) == 4:
        #     pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[self.num_envs], [pixels_shape[0] * num_frames], pixels_shape[1:]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')
        self._action_spec = self._env.action_spec()
        self._state_spec = specs.BoundedArray(shape=np.array([self.num_envs, self._env.state_space.shape[0]*num_frames]),
                                               dtype=np.float32,
                                               minimum=-np.inf,
                                               maximum=np.inf,
                                               name='state')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        # obs = np.concatenate(list(self._frames), axis=0)
        obs = np.uint8(np.concatenate(list(self._frames), axis=1)) # change the dtype of the obses
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        # pixels = self._extract_pixels(time_step)
        pixels = time_step.observation
        states = time_step.state

        for _ in range(self._num_frames):
            self._frames.append(pixels)
            self._state_frames.append(states)
        time_step = self._transform_observation(time_step)
        time_step = self._transform_state(time_step)

        return time_step

    def step(self, action):
        time_step, infos = self._env.step(action)
        # pixels = self._extract_pixels(time_step)
        pixels = time_step.observation
        states = time_step.state
        self._frames.append(pixels)
        self._state_frames.append(states)
        time_step = self._transform_observation(time_step)
        time_step = self._transform_state(time_step)
        return time_step, infos

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._obs_spec

    def state_spec(self):
        return self._state_spec

    # TODO stack the states
    def _transform_state(self, time_step):
        assert len(self._state_frames) == self._num_frames
        state = np.concatenate(list(self._state_frames), axis=1) # change the dtype of the obses
        return time_step._replace(state=state)






def make(env, frame_stack, action_repeat, num_envs, pixels_key='pixels'):
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32, num_envs)
    env = ActionRepeatWrapper(env, action_repeat, num_envs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, num_envs, pixels_key)
    # env = ExtendedTimeStepWrapper(env)
    return env