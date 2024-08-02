import os
import pickle
import numpy as np
import pandas as pd
import time
import concurrent.futures
import matplotlib.pyplot as plt
import multiprocessing
import queue
import simplejson


class Deep_Evolution_Strategy:
    def __init__(self, weights, reward_function, population_size, sigma, initial_learning_rate, decay_rate, starting_money, trend, valid_trend, new):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.local_base = 0
        self.starting_money = starting_money
        self.trend = trend
        self.valid_trend = valid_trend
        self.new = new
        self.population = []
        self.current_reward = 0
        self.max_reward = -20000000
        self.max_valid_reward = -20000000
        self.best_model_iter = 0
        self.best_valid_model_iter = 0

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def _evaluate_individual(self, weights, population_index, start_time_offset=0, end_time_offset=0, validate=False):
        weights_population = self._get_weight_from_population(weights, self.population[population_index])
        model = Model(num_of_layers=int((len(weights) - 3) / 2))
        model.weights = weights_population
        reward = self.reward_function(model, validate, start_time_offset, end_time_offset)
        return population_index, reward

    def decayed_learning_rate(self, step, decay_steps):
        return self.initial_learning_rate * 0.5 ** np.floor(step / (decay_steps // 5))

    def exponential_decayed_learning_rate(self, step, decay_steps):
        return self.initial_learning_rate * self.decay_rate ** (step / decay_steps)

    def cosine_decayed_learning_rate(self, step, decay_steps):
        return self.decay_rate + 0.5 * (self.initial_learning_rate - self.decay_rate) * (1 + np.cos(step / decay_steps * np.pi))

    def restart_cosine_decayed_learning_rate(self, step, decay_steps):
        if (step + 1) % (decay_steps // 5) == 0:
            self.local_base = step
        lowering = 1 - (self.local_base / decay_steps)
        if lowering <= 0: lowering = self.decay_rate
        return self.decay_rate + 0.5 * lowering * (self.initial_learning_rate - self.decay_rate) * (1 + np.cos((step - self.local_base) / (decay_steps // 5) * np.pi))

    def train(self, q, epoch=100, print_every=1):
        lasttime = time.time()
        learning_rate_hitory = []
        reward_history = []
        valid_reward_history = []
        weights_history = []
        learning_rate = self.initial_learning_rate

        if not self.new:
            model = Model(num_of_layers=int((len(self.weights) - 3) / 2))
            model.weights = self.weights
            rew = self.reward_function(model)
            self.current_reward = rew
            valid_rew = self.reward_function(model, True)
            reward_history.append(rew)
            valid_reward_history.append(valid_rew)
            learning_rate_hitory.append(learning_rate * 100)
            weights_history.append(np.mean(self.weights[0]) * 100)
            if rew > self.max_reward:
                self.max_reward = rew
                with open("NesModel.bin", "wb") as file:
                    pickle.dump(self.weights, file)
            if valid_rew > self.max_valid_reward:
                self.max_valid_reward = valid_rew
                with open("NesModelValid.bin", "wb") as file:
                    pickle.dump(self.weights, file)

        with concurrent.futures.ProcessPoolExecutor(10) as executor:
            """if self.new:
                current_reward = -1
                for i in range(10):
                    self.population = []
                    rewards = np.zeros(self.population_size)
                    model = Model(num_of_layers=int((len(self.weights) - 3) / 2))
                    for k in range(self.population_size):
                        x = model.init_func(self.weights[0].shape[0], self.weights[0].shape[1], self.weights[-1].shape[1])
                        self.population.append(x)

                    futures = [executor.submit(self._evaluate_individual, self.weights, k) for k in range(self.population_size)]

                    for future in concurrent.futures.as_completed(futures):
                        index, reward = future.result()
                        rewards[index] = reward

                    for j in range(len(rewards)):
                        if rewards[j] == 0:
                            rewards[j] = -1000000
                    max_reward_index = np.argmax(rewards)
                    if current_reward < rewards[max_reward_index] or current_reward == -1:
                        current_reward = rewards[max_reward_index]
                        self.weights = self.population[max_reward_index]"""

            for i in range(epoch):
                q.put(1)
                # learning_rate = self.cosine_decayed_learning_rate(i, decay_steps=epoch)
                self.population = []
                rewards = np.zeros(self.population_size)
                for k in range(self.population_size):
                    x = []
                    for w in self.weights:
                        x.append(np.random.randn(*w.shape))
                    self.population.append(x)

                if i % 2 == 0 or i % 3 == 0:
                    start_time_offset = np.random.randint(0, len(self.trend) // 1.11)
                else:
                    start_time_offset = 0

                """if i % 2 == 0 or i % 3 == 0:
                start_time_offset = np.random.randint(0, len(self.trend))
                end_time_offset = np.random.randint(0, len(self.trend))
                while start_time_offset + end_time_offset >= len(self.trend) // 1.11:
                    start_time_offset = np.random.randint(0, len(self.trend))
                    end_time_offset = np.random.randint(0, len(self.trend))"""

                """digit = i
                while digit > 10:
                    digit = digit % 10
                if digit <= 8:
                    start_time_offset = np.random.randint(0, len(self.trend))
                    end_time_offset = np.random.randint(0, len(self.trend))
                    while start_time_offset + end_time_offset >= len(self.trend) // 1.11:
                        start_time_offset = np.random.randint(0, len(self.trend))
                        end_time_offset = np.random.randint(0, len(self.trend))"""

                """digit = i
                while digit > 100:
                    digit = digit % 10
                digit = np.ceil(digit / 10)
                if digit <= 3 or digit == 5 or digit == 6 or digit == 7 or digit >= 9:
                    start_time_offset = np.random.randint(0, len(self.trend))
                    end_time_offset = 0
                    while start_time_offset + end_time_offset >= len(self.trend) // 1.11:
                        start_time_offset = np.random.randint(0, len(self.trend))
                        end_time_offset = 0"""

                futures = [executor.submit(self._evaluate_individual, self.weights, k, start_time_offset, 0) for k in range(self.population_size)]

                for future in concurrent.futures.as_completed(futures):
                    index, reward = future.result()
                    rewards[index] = reward

                for j in range(len(rewards)):
                    if rewards[j] == 0:
                        rewards[j] = -1000000
                rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
                for index, w in enumerate(self.weights):
                    A = np.array([p[index] for p in self.population])
                    self.weights[index] = (w + learning_rate / (self.population_size * self.sigma) * np.dot(A.T, rewards).T)

                if (i + 1) % print_every == 0:
                    model = Model(num_of_layers=int((len(self.weights) - 3) / 2))
                    model.weights = self.weights
                    rew = self.reward_function(model)
                    self.current_reward = rew
                    valid_rew = self.reward_function(model, True)
                    reward_history.append(rew)
                    valid_reward_history.append(valid_rew)
                    learning_rate_hitory.append(learning_rate * 100)
                    weights_history.append(np.mean(self.weights[0]) * 100)
                    print('iter', i + 1, 'reward:', round(rew, 2), 'daily:', round(rew / (len(self.trend) / 600), 2), 'valid reward:', round(valid_rew, 2), 'daily:',
                          round(valid_rew / (len(self.valid_trend) / 600), 2))
                    if rew > self.max_reward:
                        self.max_reward = rew
                        self.best_model_iter = i + 1
                        with open("NesModel.bin", "wb") as file:
                            pickle.dump(self.weights, file)
                    if valid_rew > self.max_valid_reward:
                        self.max_valid_reward = valid_rew
                        self.best_valid_model_iter = i + 1
                        with open("NesModelValid.bin", "wb") as file:
                            pickle.dump(self.weights, file)
                    IterationsLeft = 200
                    if os.path.isfile("Iterations Left.json"):
                        with open("Iterations Left.json", "r") as file:
                            IterationsLeft = simplejson.load(file)
                    with open("Iterations Left.json", "w") as file:
                        simplejson.dump(IterationsLeft - (i + 1), file, sort_keys=True, indent=4)
            q.put("NORMAL_END")
            return
            """print('Time taken to train:', round(time.time() - lasttime, 2), 'seconds. Best model iteration:', self.best_model_iter, '. Best validation model iteration:',
                  self.best_valid_model_iter)
            plt.figure(figsize=(18, 9))
            plt.plot(learning_rate_hitory, label='Learning rate multiplied by 100', c='g')
            plt.plot(reward_history, label='Reward', c='b')
            plt.plot(valid_reward_history, label='Valid reward multiplied', c='r')
            plt.plot(weights_history, label='Mean weights multiplied by 100', c='c')
            plt.legend()
            plt.show()"""


class Model:
    num_of_layers = 0

    def __init__(self, input_size=0, layer_size=0, output_size=0, num_of_layers=0, reduction=True):
        self.num_of_layers = num_of_layers
        if input_size != 0:
            self.weights = self.init_func(input_size, layer_size, output_size, reduction)

    def init_func(self, input_size, layer_size, output_size, reduction):
        return self.msra_init(input_size, layer_size, output_size, reduction)

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[1]
        feed = self.h_swish(feed)
        for i in range(self.num_of_layers):
            feed = np.dot(feed, self.weights[i + i + 2]) + self.weights[i + i + 3]
            feed = self.h_swish(feed)
        decision = np.dot(feed, self.weights[-1])
        return decision

    def relu(self, X):
        return np.maximum(X, 0)

    def relu6(self, X):
        return np.minimum(np.maximum(X, 0), 6)

    def leaky_relu(self, X, a=0.1):
        return np.maximum(X, 0) + a * np.minimum(X, 0)

    def swish(self, X):
        return X * self.sigmoid(X)

    def h_swish(self, X):
        return X * (self.relu6(X + 3) / 6)

    def elu(self, X, a=1):
        return np.maximum(X, 0) + np.minimum(0, a * (np.exp(X) - 1))

    def gelu(self, X):
        return X * self.sigmoid(1.702 * X)

    def mish(self, X):
        return X * self.tanh(self.softplus(X))

    def softmax(self, X):
        ex = np.exp(X)
        return ex / np.sum(ex, axis=1, keepdims=True)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def tanh(self, X):
        ex1 = np.exp(X)
        ex2 = np.exp(-X)
        return (ex1 - ex2) / (ex1 + ex2)

    def softplus(self, X):
        return np.log(1 + np.exp(X))

    def default_init(self, input_size, layer_size, output_size):
        return [np.random.randn(input_size, layer_size), np.random.randn(layer_size, output_size), np.random.randn(1, layer_size), ]

    def lecun_init(self, input_size, layer_size, output_size):
        return [np.random.uniform(low=-np.sqrt(3 / float(input_size)), high=np.sqrt(3 / float(input_size)), size=(input_size, layer_size)),
                np.random.uniform(low=-np.sqrt(3 / float(1)), high=np.sqrt(3 / float(1)), size=(1, layer_size)),
                np.random.uniform(low=-np.sqrt(3 / float(layer_size)), high=np.sqrt(3 / float(layer_size)), size=(layer_size, output_size))]

    def glorot_init(self, input_size, layer_size, output_size):
        return [np.random.uniform(low=-np.sqrt(6 / float(input_size + layer_size)), high=np.sqrt(6 / float(input_size + layer_size)), size=(input_size, layer_size)),
                np.random.uniform(low=-np.sqrt(6 / float(1 + layer_size)), high=np.sqrt(6 / float(1 + layer_size)), size=(1, layer_size)),
                np.random.uniform(low=-np.sqrt(6 / float(layer_size + output_size)), high=np.sqrt(6 / float(layer_size + output_size)), size=(layer_size, output_size))]

    def msra_init(self, input_size, layer_size, output_size, reduction):
        if reduction:
            out = [np.random.uniform(low=-np.sqrt(6 / float(input_size)), high=np.sqrt(6 / float(input_size)), size=(input_size, layer_size)),
                   np.random.uniform(low=-np.sqrt(6 / float(1)), high=np.sqrt(6 / float(1)), size=(1, layer_size))]
            for i in range(self.num_of_layers):
                out += [np.random.uniform(low=-np.sqrt(6 / float(layer_size / 2 ** i)), high=np.sqrt(6 / float(layer_size / 2 ** (i + 1))),
                                          size=(int(layer_size / 2 ** i), int(layer_size / 2 ** (i + 1)))),
                        np.random.uniform(low=-np.sqrt(6 / float(1)), high=np.sqrt(6 / float(1)), size=(1, int(layer_size / 2 ** (i + 1))))]

            if self.num_of_layers > 0:
                out += [np.random.uniform(low=-np.sqrt(6 / float(layer_size / (self.num_of_layers * 2))), high=np.sqrt(6 / float(layer_size / 2 ** self.num_of_layers)),
                                          size=(int(layer_size / 2 ** self.num_of_layers), output_size))]
            else:
                out += [np.random.uniform(low=-np.sqrt(6 / float(layer_size)), high=np.sqrt(6 / float(layer_size)), size=(layer_size, output_size))]
            return out
        else:
            out = [np.random.uniform(low=-np.sqrt(6 / float(input_size)), high=np.sqrt(6 / float(input_size)), size=(input_size, layer_size)),
                   np.random.uniform(low=-np.sqrt(6 / float(1)), high=np.sqrt(6 / float(1)), size=(1, layer_size))]
            for i in range(self.num_of_layers):
                out += [np.random.uniform(low=-np.sqrt(6 / float(layer_size)), high=np.sqrt(6 / float(layer_size)), size=(int(layer_size), int(layer_size))),
                        np.random.uniform(low=-np.sqrt(6 / float(1)), high=np.sqrt(6 / float(1)), size=(1, int(layer_size)))]
            else:
                out += [np.random.uniform(low=-np.sqrt(6 / float(layer_size)), high=np.sqrt(6 / float(layer_size)), size=(layer_size, output_size))]
            return out


class Agent:
    def __init__(self, model, starting_money, population_size, sigma, initial_learning_rate, decay_rate, skip, new, min_increment):
        self.window_size = len(model.weights[0])
        self.skip = skip
        self.close = pd.read_csv('dataTCSG.csv', sep=';').Close.values.tolist()
        self.i_close = self.preprocess(self.close)
        self.valid_close = pd.read_csv('validTCSG.csv', sep=';').Close.values.tolist()
        self.valid_i_close = self.preprocess(self.valid_close)
        self.date = pd.read_csv('dataTCSG.csv', sep=';').Date.values.tolist()
        self.valid_date = pd.read_csv('validTCSG.csv', sep=';').Date.values.tolist()
        self.high = pd.read_csv('dataTCSG.csv', sep=';').High.values.tolist()
        self.i_high = self.preprocess(self.high)
        self.valid_high = pd.read_csv('validTCSG.csv', sep=';').High.values.tolist()
        self.valid_i_high = self.preprocess(self.valid_high)
        self.low = pd.read_csv('dataTCSG.csv', sep=';').Low.values.tolist()
        self.i_low = self.preprocess(self.low)
        self.valid_low = pd.read_csv('validTCSG.csv', sep=';').Low.values.tolist()
        self.valid_i_low = self.preprocess(self.valid_low)
        self.volume = pd.read_csv('dataTCSG.csv', sep=';').Volume.values.tolist()
        self.i_volume = self.preprocess(self.volume, True)
        self.valid_volume = pd.read_csv('validTCSG.csv', sep=';').Volume.values.tolist()
        self.valid_i_volume = self.preprocess(self.valid_volume, True)
        self.starting_money = starting_money
        self.min_increment = min_increment
        self.es = Deep_Evolution_Strategy(model.weights, self.get_reward, population_size, sigma, initial_learning_rate, decay_rate, starting_money, self.close, self.valid_close,
                                          new)

    """
    def get_state(self, data, t, n):
        d = t - n + 1
        block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[: t + 1]
        res = []
        for i in range(n - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])
    """

    def preprocess(self, data, divide=False):
        res = []
        for i in range(len(data) - 1):
            if divide:
                diff = (data[i + 1] - data[i]) / 20000
            else:
                diff = data[i + 1] - data[i]
            if diff > 3:
                res.append(3)
            elif diff < -3:
                res.append(-3)
            elif diff == 0:
                res.append(1e-7)
            else:
                res.append(diff)
        return res

    def get_state(self, data, t, n, inv, currentPrice, day_limit_counter, high, low, volume):
        d = t - n + 1
        invNum = int((n - 1) / 10 * 2)
        # block = data[d + invNum * 2:t]
        piece = int((n - 1 - invNum) / 4)
        block = data[d + invNum + piece * 3:t]
        block.extend(high[d + invNum + piece * 3:t - 1])
        block.append(1e-7)
        block.extend(low[d + invNum + piece * 3:t - 1])
        block.append(1e-7)
        block.extend(volume[d + invNum + piece * 3:t - 1])
        block.append(1e-7)
        for i in range(invNum):
            if inv:
                if (currentPrice - inv[0]) / 4 > 3:
                    block.append(3)
                elif (currentPrice - inv[0]) / 4 < -3:
                    block.append(-3)
                else:
                    block.append((currentPrice - inv[0]) / 4 + 1e-7)
            else:
                block.append(1e-7)
        return block

    def act(self, sequence, model):
        decision = model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def get_reward(self, model, validate=False, start_time_offset=0, end_time_offset=0):
        close = self.close
        i_close = self.i_close
        date = self.date
        i_high = self.i_high
        i_low = self.i_low
        i_volume = self.i_volume
        if validate:
            close = self.close[-self.window_size:] + self.valid_close
            i_close = self.i_close[-self.window_size:] + self.valid_i_close
            date = self.date[-self.window_size:] + self.valid_date
            i_high = self.i_high[-self.window_size:] + self.valid_i_high
            i_low = self.i_low[-self.window_size:] + self.valid_i_low
            i_volume = self.i_volume[-self.window_size:] + self.valid_i_volume

        mean = np.mean(close)

        starting_money = self.starting_money
        current_money = starting_money

        inventory = []

        since_last_action = 0

        day_limit_counter = 0

        start_time = self.window_size + start_time_offset
        end_time = len(close) - 1 - end_time_offset

        state = self.get_state(i_close, start_time, self.window_size + 1, inventory, close[start_time], day_limit_counter, i_high, i_low, i_volume)

        for t in range(start_time, end_time, self.skip):
            if date[t] == "10:00:00":
                day_limit_counter = 0
            if day_limit_counter < 100 and (date[t] < "18:40:00" or date[t] > "19:00:00"):
                action = self.act(state, model)

                if action == 1 and current_money >= close[t] and not inventory:
                    current_money -= close[t]
                    inventory.append(close[t])
                    since_last_action = 0
                    day_limit_counter += 1
                elif action == 2 and inventory:
                    current_money += close[t] - self.min_increment
                    inventory.pop(0)
                    since_last_action = 0
                    day_limit_counter += 1
                # elif action == 1 and inventory and not validate:
                # current_money -= mean / 100
                # elif action == 2 and not inventory and not validate:
                # current_money -= mean / 100

                # if since_last_action >= 240 and not validate:
                # current_money -= mean / 1000

                since_last_action += 1

                if inventory:
                    state = self.get_state(i_close, t + 1, self.window_size + 1, inventory, close[t + 1] - self.min_increment, day_limit_counter, i_high, i_low, i_volume)
                else:
                    state = self.get_state(i_close, t + 1, self.window_size + 1, inventory, close[t + 1], day_limit_counter, i_high, i_low, i_volume)

        if inventory:
            current_money += close[end_time] - self.min_increment
        total_gains = current_money - starting_money
        invest = total_gains / mean * 100
        return invest

    def fit(self, q, iterations, checkpoint):
        self.es.train(q, iterations, print_every=checkpoint)


def pinger(q):
    while True:
        try:
            if q.get(timeout=300.0) == "NORMAL_END":
                q.put("NORMAL_END")
        except queue.Empty:
            q.put("EMERGENCY_END")


def mainQueue(agent, iterations):
    q = multiprocessing.Queue()
    mainProcess = multiprocessing.Process(target=agent.fit, args=(q, iterations, 10,))
    pingerProcess = multiprocessing.Process(target=pinger, args=(q,))
    pingerProcess.daemon = True
    mainProcess.start()
    pingerProcess.start()
    while True:
        msg = q.get()
        if msg == "EMERGENCY_END" or msg == "NORMAL_END":
            mainProcess.terminate()
            time.sleep(2)
            if not mainProcess.is_alive():
                mainProcess.join()
                q.close()
                break
    if msg == "EMERGENCY_END":
        print(time.strftime("%H:%M:%S") + " Program not responding - restart initiated.")
        return False
    return True


if __name__ == '__main__':
    print(time.strftime("%H:%M:%S"), " Nightly model training initiated.")

    IterationsLeft = 200
    with open("Iterations Left.json", "w") as file:
        simplejson.dump(IterationsLeft, file, sort_keys=True, indent=4)

    while True:
        if os.path.isfile("Iterations Left.json"):
            with open("Iterations Left.json", "r") as file:
                IterationsLeft = simplejson.load(file)

        CONTINUE = True
        model = Model(input_size=240, layer_size=120, output_size=3, num_of_layers=4, reduction=True)

        new = True
        if CONTINUE:
            if os.path.isfile("NesModelValid.bin"):
                with open("NesModelValid.bin", "rb") as file:
                    model.weights = pickle.load(file)
                    new = False

        agent = Agent(model=model, starting_money=100000, population_size=150, sigma=0.1, initial_learning_rate=0.03, decay_rate=0.01, skip=1, new=new, min_increment=0.5)
        if not mainQueue(agent, IterationsLeft):
            continue
        else:
            break

"""
            if action == 1 and current_money >= close[t] and not inventory:
                history = 120
                if t - day_start - history < 0:
                    history = t - day_start 
                if history == 0:
                    avg_price = close[t]
                else:
                    avg_price = np.sum(close[t - history:t]) / history
                if (avg_price - close[t]) / avg_price * 100 >= 0.05:
                    current_money -= close[t]
                    inventory.append(close[t])
            elif action == 2 and inventory and ((close[t] - inventory[0]) / inventory[0] * 100 >= 0.15):
                current_money += close[t] * 0.9992
                inventory.pop(0)
            """
