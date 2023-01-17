# raise ValueError()
import time
import gym
import tensorflow as tf
import numpy as np

tf.compat.v1.reset_default_graph() # pour supprimer les variables tensorflow d'avant
tf.compat.v1.disable_eager_execution() # nécessaire pour utiliser compat.v1

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state = env.reset()
prev_screen = env.render(mode='rgb_array')

# tf_input_size = env.observation_space.n
tf_input_size = 9
tf_output_size = env.action_space.n
tf_hidden_size = (tf_input_size + tf_output_size)//2

tf_inputs = tf.compat.v1.placeholder(tf.float32, [None, tf_input_size])
tf_next_q = tf.compat.v1.placeholder(tf.float32, [None, tf_output_size])

weight_initer = tf.compat.v1.truncated_normal_initializer(
    mean=.00, stddev=20)
biases_initer = tf.compat.v1.truncated_normal_initializer(
    mean=5, stddev=20)

tf_weights_1 = tf.compat.v1.get_variable(
    "tf_weights_1", [tf_input_size, tf_hidden_size], initializer=weight_initer)
tf_biases_1 = tf.compat.v1.get_variable(
    "tf_biases_1", [tf_hidden_size], initializer=biases_initer)
tf_outputs_1 = tf.nn.relu(tf.matmul(tf_inputs, tf_weights_1) + tf_biases_1)

tf_weights_out = tf.compat.v1.get_variable(
    "tf_weights_out", [tf_hidden_size, tf_output_size], initializer=weight_initer)
tf_biases_out = tf.compat.v1.get_variable(
    "tf_biases_out", [tf_output_size], initializer=biases_initer)
tf_outputs = tf.matmul(tf_outputs_1, tf_weights_out) + tf_biases_out

tf_action = tf.argmax(input=tf_outputs, axis=1)
tf_loss = tf.reduce_sum(input_tensor=tf.square(tf_outputs - tf_next_q))
tf_optimize = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.0001).minimize(tf_loss)

sess = tf.compat.v1.InteractiveSession()
initializer = tf.compat.v1.global_variables_initializer()
sess.run(initializer)

desc3=["XXXXXX",
       "XSFFFX", 
       "XFHFHX", 
       "XFFFHX", 
       "XHFFGX",
       "XXXXXX"]


for i in range(6):
    desc3[i] = desc3[i].replace("X","0")
    desc3[i] = desc3[i].replace("S","1")
    desc3[i] = desc3[i].replace("F","2")
    desc3[i] = desc3[i].replace("H","3")
    desc3[i] = desc3[i].replace("G","4")
    
for episode in range(100):
    env.reset()
    for i in range(50):

        input = np.zeros((3,3),dtype="U1")
        action = env.action_space.sample()
        x, y = state//4, state % 4
        x += 1
        y += 1

        for u in [0, 1, 2]:
            for v in [0, 1, 2]:
                input[u][v]=2
                input[u][v] = desc3[x+(u-1)][y+(v-1)]
    
        action = sess.run([tf_action], feed_dict={
                          tf_inputs: np.reshape(input, (1, 9))})[0][0]

        time.sleep(1)
        new_state, reward, done, info = env.step(action)
        screen = env.render(mode='rgb_array')

        state = new_state
        if done:
            break
env.close()
