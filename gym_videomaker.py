from gym.wrappers.record_video import RecordVideo

env = gym.make('Acrobot-v1', render_mode="rgb_array")
env = RecordVideo(env, 'saved_networks/video', episode_trigger = lambda x: x % 2 == 0)
state=env.reset()
for i in range(500):
    if len(state)==2:
        state=state[0]
    state,  reward, done, info,_  = env.step(np.argmax(model(tf.expand_dims(state,0)).numpy()[0]))
env.close()
