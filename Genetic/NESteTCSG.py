import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


class Model:
    num_of_layers = 0

    def __init__(self, weights):
        self.weights = weights
        self.num_of_layers = int((len(weights) - 3) / 2)

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[1]
        feed = self.h_swish(feed)
        for i in range(self.num_of_layers):
            feed = np.dot(feed, self.weights[i + i + 2]) + self.weights[i + i + 3]
            feed = self.h_swish(feed)
        decision = np.dot(feed, self.weights[-1])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def relu6(self, X):
        return np.minimum(np.maximum(X, 0), 6)

    def h_swish(self, X):
        return X * (self.relu6(X + 3) / 6)

    def relu(self, X):
        return np.maximum(X, 0)

    def softmax(self, X):
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))


class Agent:
    def __init__(self, model, starting_money, skip, min_increment):
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
        self.model = model
        self.starting_money = starting_money
        self.min_increment = min_increment

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

    def get_state(self, data, t, n, inv, currentPrice, high, low, volume):
        d = t - n + 1
        invNum = int((n - 1) / 10)
        #block = data[d + invNum:t]
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

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def buy(self):
        start_time = self.window_size
        """close = self.close[-self.window_size:] + self.valid_close
        i_close = self.i_close[-self.window_size:] + self.valid_i_close
        date = self.date[-self.window_size:] + self.valid_date
        i_high = self.i_high[-self.window_size:] + self.valid_i_high
        i_low = self.i_low[-self.window_size:] + self.valid_i_low
        i_volume = self.i_volume[-self.window_size:] + self.valid_i_volume"""
        close = self.close + self.valid_close
        i_close = self.i_close + self.valid_i_close
        date = self.date + self.valid_date
        i_high = self.i_high + self.valid_i_high
        i_low = self.i_low + self.valid_i_low
        i_volume = self.i_volume + self.valid_i_volume
        """close = self.close
        i_close = self.i_close
        date = self.date
        i_high = self.i_high
        i_low = self.i_low
        i_volume = self.i_volume"""
        starting_money = self.starting_money
        current_money = starting_money
        states_sell = []
        states_buy = []
        inventory = []

        since_last_action = 1e-7

        day_limit_counter = 0
        total_sales = 0
        negative_sales = 0
        zero_sales = 0
        positive_sales = 0

        state = self.get_state(i_close, start_time, self.window_size + 1, inventory, close[start_time], i_high, i_low, i_volume)

        for t in range(start_time, len(close) - 1, self.skip):
            if date[t] == "10:00:00":
                day_limit_counter = 0
                print("New day starts")
            if day_limit_counter < 100 and (date[t] < "18:40:00" or date[t] > "19:00:00"):
                action = self.act(state)

                if action == 1 and current_money >= close[t] and not inventory:
                    current_money -= close[t]
                    inventory.append(close[t])
                    states_buy.append(t)
                    since_last_action = 1e-7
                    day_limit_counter += 1
                    print('minute %d, local time %s: buy at price %f, total balance %f' % (t, date[t], close[t], current_money))

                elif action == 2 and inventory:
                    bought_price = inventory.pop(0)
                    current_money += close[t] - self.min_increment
                    states_sell.append(t)
                    since_last_action = 1e-7
                    day_limit_counter += 1
                    total_sales += 1
                    if close[t] - self.min_increment - bought_price > 0:
                        positive_sales += 1
                    elif close[t] - self.min_increment - bought_price < 0:
                        negative_sales += 1
                    else:
                        zero_sales += 1
                    try:
                        invest = ((close[t] - self.min_increment - bought_price) / bought_price) * 100
                    except:
                        invest = 0
                    print('minute %d, local time %s: sell at price %f, investment %f %%, total balance %f,' % (t, date[t], close[t] - self.min_increment, invest, current_money))

                since_last_action += 1

            state = self.get_state(i_close, t + 1, self.window_size + 1, inventory, close[t + 1], i_high, i_low, i_volume)

        if inventory:
            current_money += close[len(close) - 1] - self.min_increment
        total_gains = current_money - starting_money
        invest = total_gains / np.mean(close) * 100
        # invest = total_gains / starting_money * 100

        print("Test results: Gain - ", total_gains, " Profit - ", round(invest, 2), "% Daily profit - ", round(invest / (len(close) / 600), 2), "%")
        print("Total number of sales - ", total_sales, ", average number of sales per day - ", round(total_sales / (len(close) / 600), 2))
        print("Number of positive sales - ", positive_sales, ", as a proportion of total sales - ", round(positive_sales / total_sales * 100, 2), "%")
        print("Number of zero sales - ", zero_sales, ", as a proportion of total sales - ", round(zero_sales / total_sales * 100, 2), "%")
        print("Number of negative sales - ", negative_sales, ", as a proportion of total sales - ", round(negative_sales / total_sales * 100, 2), "%")
        plt.figure(figsize=(18, 9))
        plt.plot(close, label='true close', c='g')
        plt.plot(close, 'X', label='predict buy', markevery=states_buy, c='b')
        plt.plot(close, 'o', label='predict sell', markevery=states_sell, c='r')
        plt.legend()
        plt.show()


if os.path.isfile("NesModelValid.bin"):
    with open("NesModelValid.bin", "rb") as file:
        model = Model(pickle.load(file))

agent = Agent(model=model, starting_money=3500, skip=1, min_increment=0.5)
agent.buy()
