import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

MAX_BACKLOG = 1
HISTORY_HOURS = 3.5
SELLING_MARGIN = 0.3
BUYING_MARGIN = 0.9
ALLOWED_TRAIL_SELL_PERCENT = 35
ALLOWED_TRAIL_BUY_PERCENT = 15
BACKLOG_SELLING_MARGIN = 0.05
STRICT_SELLING_MARGIN = 0.2
STRICT_BUYING_MARGIN = 0

DROP_OFF_START = 0.65
DROP_OFF_LIMIT = 10
DROP_OFF_TIME = 11

MIN_INCREMENT = 0.5

""" TCSG INCREMENT
MAX_BACKLOG = 1
HISTORY_HOURS = 3.5
SELLING_MARGIN = 0.3
BUYING_MARGIN = 0.9
ALLOWED_TRAIL_SELL_PERCENT = 35
ALLOWED_TRAIL_BUY_PERCENT = 15
BACKLOG_SELLING_MARGIN = 0.05
STRICT_SELLING_MARGIN = 0.2
STRICT_BUYING_MARGIN = 0
INITIAL_QUANTITY = 333

DROP_OFF_START = 0.65
DROP_OFF_LIMIT = 10
DROP_OFF_TIME = 15

MIN_INCREMENT = 0.5
"""

""" TCSG NO INCREMENT
MAX_BACKLOG = 1
HISTORY_HOURS = 3.5
SELLING_MARGIN = 0.2
BUYING_MARGIN = 0.85
ALLOWED_TRAIL_SELL_PERCENT = 35
ALLOWED_TRAIL_BUY_PERCENT = 15
BACKLOG_SELLING_MARGIN = 0.15
STRICT_SELLING_MARGIN = 0.2
STRICT_BUYING_MARGIN = 0
INITIAL_QUANTITY = 333

DROP_OFF_START = 0.65
DROP_OFF_LIMIT = 10
DROP_OFF_TIME = 12
"""

""" SAFE NO NEG
MAX_BACKLOG = 1
HISTORY_HOURS = 6
SELLING_MARGIN = 0.25
BUYING_MARGIN = 0.35
ALLOWED_TRAIL_SELL_PERCENT = -1
ALLOWED_TRAIL_BUY_PERCENT = -1
BACKLOG_SELLING_MARGIN = 0.15
STRICT_SELLING_MARGIN = 0.2
STRICT_BUYING_MARGIN = 0
INITIAL_QUANTITY = 333
"""

""" SAFE 2
MAX_BACKLOG = 1
HISTORY_HOURS = 6
SELLING_MARGIN = 0.25
BUYING_MARGIN = 0.5
ALLOWED_TRAIL_SELL_PERCENT = 15
ALLOWED_TRAIL_BUY_PERCENT = 20
BACKLOG_SELLING_MARGIN = 0.15
STRICT_SELLING_MARGIN = 0.2
STRICT_BUYING_MARGIN = 0
INITIAL_QUANTITY = 333
"""

""" SAFE
MAX_BACKLOG = 1
HISTORY_HOURS = 4.5
SELLING_MARGIN = 0.25
BUYING_MARGIN = 0.3
ALLOWED_TRAIL_SELL_PERCENT = 15
ALLOWED_TRAIL_BUY_PERCENT = 20
BACKLOG_SELLING_MARGIN = 0.15
STRICT_SELLING_MARGIN = 0.2
STRICT_BUYING_MARGIN = 0
INITIAL_QUANTITY = 333
"""

""" RISKY
MAX_BACKLOG = 1
HISTORY_HOURS = 1.15
SELLING_MARGIN = 0.34
BUYING_MARGIN = 0.1
ALLOWED_TRAIL_SELL_PERCENT = 20
ALLOWED_TRAIL_BUY_PERCENT = 20
BACKLOG_SELLING_MARGIN = 0.22
STRICT_SELLING_MARGIN = 0.34
STRICT_BUYING_MARGIN = 0
INITIAL_QUANTITY = 333
"""


class Agent:
    VARIABLES = {'TOTAL_PROFIT_UNIT': 0}

    def __init__(self, close, valid_close, starting_money):
        self.close = close
        self.valid_close = valid_close
        self.starting_money = starting_money

    def getAveragePrice(self, close, t):
        history = round(60 * HISTORY_HOURS)
        if t - history < 0:
            history = t
        if history == 0:
            return close[t]
        else:
            return np.sum(close[t - history:t]) / history

    def getQuantity(self, curPrice):
        return math.floor((self.starting_money + (self.VARIABLES['TOTAL_PROFIT_UNIT'] * 0.9)) / curPrice)

    def buy(self):
        close = self.close + self.valid_close
        starting_money = self.starting_money
        current_money = starting_money
        states_sell = []
        states_buy = []
        inventory = []
        BACKLOG = []

        minimalPrice = 100000000000000
        maximalPrice = 0
        startingPrice = 0

        lastPrice = 0

        drop_off_num = 0

        for t in range(0, len(close) - 1):
            lastPrice = close[t]

            if BACKLOG:
                for i in range(len(BACKLOG) - 1, -1, -1):
                    if (close[t]-MIN_INCREMENT) >= (BACKLOG[i][0] * (BACKLOG_SELLING_MARGIN / 100 + 1)):
                        bought_price = BACKLOG[i]
                        del BACKLOG[i]
                        current_money += (close[t]-MIN_INCREMENT) * bought_price[2]
                        self.VARIABLES['TOTAL_PROFIT_UNIT'] += ((close[t]-MIN_INCREMENT) * bought_price[2]) - (bought_price[0] * bought_price[2])
                        states_sell.append(t)
                        try:
                            invest = (((close[t]-MIN_INCREMENT) - bought_price[0]) / bought_price[0]) * 100
                        except:
                            invest = 0
                        print('day %d, sell backlog at price %f, investment %f %%, total balance %f,' % (t, (close[t]-MIN_INCREMENT), invest, current_money))
                        drop_off_num = 0
                    else:
                        cur_loss = ((close[t]-MIN_INCREMENT) - BACKLOG[i][0]) / (close[t]-MIN_INCREMENT) * -100
                        if DROP_OFF_LIMIT >= cur_loss >= DROP_OFF_START:
                            if drop_off_num >= DROP_OFF_TIME:
                                bought_price = BACKLOG[i]
                                del BACKLOG[i]
                                current_money += (close[t]-MIN_INCREMENT) * bought_price[2]
                                self.VARIABLES['TOTAL_PROFIT_UNIT'] += ((close[t]-MIN_INCREMENT) * bought_price[2]) - (bought_price[0] * bought_price[2])
                                states_sell.append(t)
                                try:
                                    invest = (((close[t]-MIN_INCREMENT) - bought_price[0]) / bought_price[0]) * 100
                                except:
                                    invest = 0
                                print('day %d, drop off sell at price %f, investment %f %%, total balance %f,' % (t, (close[t]-MIN_INCREMENT), invest, current_money))
                                drop_off_num = 0
                            else:
                                drop_off_num += 1

            if not inventory and len(BACKLOG) < MAX_BACKLOG:
                if startingPrice == 0:
                    avg_price = self.getAveragePrice(close, t)
                    if (avg_price - close[t]) / avg_price * 100 >= BUYING_MARGIN:
                        startingPrice = close[t]
                else:
                    if minimalPrice == 100000000000000 and close[t] > startingPrice:
                        startingPrice = 0
                    elif minimalPrice >= close[t]:
                        minimalPrice = close[t]
                    elif (((startingPrice - minimalPrice) - (startingPrice - close[t])) / (startingPrice - minimalPrice + 1e-7) * 100) > ALLOWED_TRAIL_BUY_PERCENT:
                        if close[t] <= startingPrice * (1 - STRICT_BUYING_MARGIN / 100):
                            quantity = self.getQuantity(close[t])
                            current_money -= close[t] * quantity
                            inventory.append([close[t], t, quantity])
                            states_buy.append(t)
                            print('day %d: buy at price %f, total balance %f' % (t, close[t], current_money))
                            startingPrice = 0
                            minimalPrice = 100000000000000
                        else:
                            startingPrice = 0
                            minimalPrice = 100000000000000

            elif inventory:
                if startingPrice == 0:
                    if ((close[t]-MIN_INCREMENT) - inventory[0][0]) / inventory[0][0] * 100 >= SELLING_MARGIN:
                        startingPrice = inventory[0][0]
                    elif ((t - inventory[0][1] >= 60) and (len(BACKLOG) < MAX_BACKLOG - 1) and (inventory[0][0] > self.getAveragePrice(close, t) * (1 + BUYING_MARGIN / 100))) or (
                            t - inventory[0][1] >= 500):
                        BACKLOG.append(inventory[0])
                        inventory.pop(0)
                        print('day %d, order moved to backlog,' % t)

                    if inventory and startingPrice == 0:
                        cur_loss = ((close[t]-MIN_INCREMENT) - inventory[0][0]) / (close[t]-MIN_INCREMENT) * -100
                        if DROP_OFF_LIMIT >= cur_loss >= DROP_OFF_START:
                            if drop_off_num >= DROP_OFF_TIME:
                                bought_price = inventory.pop(0)
                                current_money += (close[t]-MIN_INCREMENT) * bought_price[2]
                                self.VARIABLES['TOTAL_PROFIT_UNIT'] += ((close[t]-MIN_INCREMENT) * bought_price[2]) - (bought_price[0] * bought_price[2])
                                states_sell.append(t)
                                try:
                                    invest = (((close[t]-MIN_INCREMENT) - bought_price[0]) / bought_price[0]) * 100
                                except:
                                    invest = 0
                                print('day %d, drop off sell at price %f, investment %f %%, total balance %f,' % (t, (close[t]-MIN_INCREMENT), invest, current_money))
                                drop_off_num = 0
                            else:
                                drop_off_num += 1
                else:
                    if maximalPrice == 0 and (close[t]-MIN_INCREMENT) < startingPrice:
                        startingPrice = 0
                    elif (close[t]-MIN_INCREMENT) >= maximalPrice:
                        maximalPrice = (close[t]-MIN_INCREMENT)
                    elif (((maximalPrice - startingPrice) - ((close[t]-MIN_INCREMENT) - startingPrice)) / (maximalPrice - startingPrice + 1e-7) * 100) > ALLOWED_TRAIL_SELL_PERCENT:
                        if (close[t]-MIN_INCREMENT) >= startingPrice * (STRICT_SELLING_MARGIN / 100 + 1):
                            bought_price = inventory.pop(0)
                            current_money += (close[t]-MIN_INCREMENT) * bought_price[2]
                            self.VARIABLES['TOTAL_PROFIT_UNIT'] += ((close[t]-MIN_INCREMENT) * bought_price[2]) - (bought_price[0] * bought_price[2])
                            states_sell.append(t)
                            try:
                                invest = (((close[t]-MIN_INCREMENT) - bought_price[0]) / bought_price[0]) * 100
                            except:
                                invest = 0
                            print('day %d, sell at price %f, investment %f %%, total balance %f,' % (t, (close[t]-MIN_INCREMENT), invest, current_money))
                            startingPrice = 0
                            maximalPrice = 0
                            drop_off_num = 0
                        else:
                            startingPrice = 0
                            maximalPrice = 0

        if inventory:
            current_money += (lastPrice-MIN_INCREMENT) * inventory[0][2]
        for purchase in BACKLOG:
            current_money += (lastPrice-MIN_INCREMENT) * purchase[2]
        total_gains = current_money - starting_money
        invest = total_gains / starting_money * 100

        print("Test results: Gain - ", total_gains, " Profit - ", invest, "% Daily profit - ", invest / (len(close) / 500), "%")
        print(len(states_buy)," ", invest, "%")
        plt.figure(figsize=(18, 9))
        plt.plot(close, label='true close', c='g')
        plt.plot(close, 'X', label='predict buy', markevery=states_buy, c='b')
        plt.plot(close, 'o', label='predict sell', markevery=states_sell, c='r')
        plt.legend()
        plt.show()


        return [len(states_buy), invest]



data = pd.read_csv('dataTCSG.csv', sep=';')
close = data.Close.values.tolist()

valid = pd.read_csv('validTCSG.csv', sep=';')
valid_close = valid.Close.values.tolist()

agent = Agent(close=close, valid_close=valid_close, starting_money=1100000 * MAX_BACKLOG)
agent.buy()

"""RESULTS = []
RESULTS.append([])
RESULTS[0].append(0)
for i in range(1, 30):
    DROP_OFF_TIME = i
    agent.starting_money = 1100000 * MAX_BACKLOG
    RESULTS[0].append(agent.buy())
    agent.VARIABLES['TOTAL_PROFIT_UNIT'] = 0
    agent.VARIABLES['QUANTITY'] = INITIAL_QUANTITY

for i in range(1, len(RESULTS[0])):
    print(i, " ", RESULTS[0][i][0], " ", RESULTS[0][i][1], "%")"""

