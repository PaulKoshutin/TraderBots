from datetime import timedelta
import time
import simplejson
from decimal import *
import os
import math
import keyboard
import multiprocessing
import queue
import pickle
import pandas as pd
import numpy as np
import subprocess

from tinkoff.invest import Client, CandleInterval, InvestError
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from tinkoff.invest.grpc.common_pb2 import SECURITY_TRADING_STATUS_NORMAL_TRADING, SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING
from tinkoff.invest.grpc.instruments_pb2 import InstrumentIdType
from tinkoff.invest.grpc.orders_pb2 import *
from tinkoff.invest.utils import now, quotation_to_decimal, decimal_to_quotation

SAND_TOKEN = 't.'  # Add own token
REAL_TOKEN = 't.'  # Add own token

SAND_ACCOUNT_ID = ""  # Add own ID
REAL_ACCOUNT_ID = ""  # Add own ID

INSTRUMENT_ID = 'eed9621b-6412-4f4b-a166-758882cc7a4c'

TARGET = INVEST_GRPC_API
TOKEN = REAL_TOKEN
ACCOUNT_ID = REAL_ACCOUNT_ID

MAX_BACKLOG = 1
HISTORY_HOURS = 1.15
SLEEP_TIME = 1
TRY_AGAIN_TIME_SHORT = 2
TRY_AGAIN_TIME_LONG = 30
SELLING_MARGIN = 0.34
BUYING_MARGIN = 0.1
ALLOWED_TRAIL_SELL_PERCENT = 20
ALLOWED_TRAIL_BUY_PERCENT = 20
BACKLOG_SELLING_MARGIN = 0.22
STRICT_SELLING_MARGIN = 0.34
STRICT_BUYING_MARGIN = 0
INITIAL_QUANTITY = 3
MINIMAL_SALE_MARGIN = 0
RISKY_SALE_COUNTER_MAX = 20
BUY_WAIT_LIMIT = 60

BOUGHT = []
VARIABLES = {'ORDER_ID': 0, 'TOTAL_PROFIT_PERCENT': 0, 'TOTAL_PROFIT_UNIT': 0, 'OLD_QUANTITY': INITIAL_QUANTITY, 'NEW_QUANTITY': INITIAL_QUANTITY + 1,
             'ALLOWED_ORDERS_WITH_NEW_QUANTITY': 0, 'DECIMAL_POINTS': 2, 'MIN_PRICE_INCREMENT': 0, 'DATE': "", 'LOT_SIZE': 0, 'MONTHLY_PROFIT_PERCENT': 0, 'MONTHLY_PROFIT_UNIT': 0, 'MONTH': ""}
BACKLOG_IDS = []
DAILY_VARIABLES = {'DAILY_PROFIT_PERCENT': 0, 'DAILY_PROFIT_UNIT': 0, 'DAILY_BUYS': 0, 'DAILY_SALES': 0, 'DAY_START_BACKLOG': 0, 'RISKY_SALE_COUNTER': 0, 'DAY_LIMIT_COUNTER': 0, 'BUY_WAIT_COUNTER': 0}
SPECIAL_VARIABLES = {'BACKLOG_SET_FOR_SALE': False, 'FIRST_ORDER': True}


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
    def __init__(self, model):
        self.window_size = len(model.weights[0])
        self.model = model
        self.prevDay = None
        while self.prevDay is None:
            self.prevDay = self.getPrevDay()

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

    def get_state(self, data, windowSize, inv, currentPrice, day_limit_counter, high, low, volume):
        invNum = int(windowSize / 10)
        block = data[invNum:]
        """piece = int((windowSize - invNum) / 4)
        block = data[invNum + piece * 3:]
        block.extend(high[invNum + piece * 3:])
        block.append(1e-7)
        block.extend(low[invNum + piece * 3:])
        block.append(1e-7)
        block.extend(volume[invNum + piece * 3:])
        block.append(1e-7)"""
        for i in range(invNum):
            if inv:
                if (currentPrice - float(inv[0][0])) / 4 > 3:
                    block.append(3)
                elif (currentPrice - float(inv[0][0])) / 4 < -3:
                    block.append(-3)
                else:
                    block.append((currentPrice - float(inv[0][0])) / 4 + 1e-7)
            else:
                block.append(1e-7)
        return block

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def getPrevDay(self):
        with Client(TOKEN, target=TARGET) as client:
            candles = None
            if time.strftime("%a") != "Mon":
                prevDayStartH = 24 + (int(time.strftime("%H")) - 10)
            else:
                prevDayStartH = 72 + (int(time.strftime("%H")) - 10)
            prevDayStartM = int(time.strftime("%M"))
            fromTime = now() - timedelta(hours=prevDayStartH, minutes=prevDayStartM)
            toTime = now() - timedelta(hours=prevDayStartH - 14, minutes=prevDayStartM)
            for x in range(0, 2):  # try 2 times
                try:
                    candles = client.market_data.get_candles(from_=fromTime, to=toTime, interval=CandleInterval.CANDLE_INTERVAL_1_MIN, instrument_id=INSTRUMENT_ID).candles
                except InvestError as error:
                    print("Get previous day prices. Exception in get_candles:", error)
                    time.sleep(TRY_AGAIN_TIME_SHORT)  # wait for 1 second before trying again
                else:
                    break
            prices = []
            highs = []
            lows = []
            volumes = []
            if candles is not None:
                for candle in candles:
                    prices.append(float(quotation_to_decimal(candle.close)))
                    highs.append(float(quotation_to_decimal(candle.high)))
                    lows.append(float(quotation_to_decimal(candle.low)))
                    volumes.append(float(candle.volume))
                return prices, highs, lows, volumes
        time.sleep(TRY_AGAIN_TIME_LONG)
        return None

    def getHistory(self):
        with Client(TOKEN, target=TARGET) as client:
            candles = None
            prices, highs, lows, volumes = self.prevDay
            curTime = int(time.strftime("%H")) * 60 + int(time.strftime("%M")) - 10 * 60
            for x in range(0, 2):  # try 2 times
                try:
                    candles = client.market_data.get_candles(from_=now() - timedelta(minutes=curTime), to=now(), interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
                                                             instrument_id=INSTRUMENT_ID).candles
                except InvestError as error:
                    print("Get price history. Exception in get_candles:", error)
                    time.sleep(TRY_AGAIN_TIME_SHORT)  # wait for 1 second before trying again
                else:
                    break
            if candles is not None:
                for candle in candles:
                    prices.append(float(quotation_to_decimal(candle.close)))
                    highs.append(float(quotation_to_decimal(candle.high)))
                    lows.append(float(quotation_to_decimal(candle.low)))
                    volumes.append(float(candle.volume))
                return prices[-self.window_size:], highs[-self.window_size:], lows[-self.window_size:], volumes[-self.window_size:]
        time.sleep(TRY_AGAIN_TIME_LONG)
        return None

    def predict(self, inventory=[], quantity=0):
        close, high, low, volume = self.getHistory()
        if close is None:
            return False
        currentPrice = getCurrentPrice()
        if currentPrice is None:
            return False
        currentPrice = float(currentPrice)
        close.append(currentPrice)
        i_close = self.preprocess(close)
        i_high = self.preprocess(high)
        i_low = self.preprocess(low)
        i_volume = self.preprocess(volume)

        state = self.get_state(i_close, self.window_size, inventory, currentPrice, DAILY_VARIABLES['DAY_LIMIT_COUNTER'], i_high, i_low, i_volume)
        action = self.act(state)

        if action == 1 and not inventory:
            if not buy(currentPrice):
                print(time.strftime("%H:%M:%S") + " Purchasing operation failed - restart initiated.")
                return False
            else:
                print(time.strftime("%H:%M:%S") + " Purchase initiated. " + "Current price: " + str(currentPrice))
                return True


        elif action == 2 and inventory:
            if (currentPrice >= float(inventory[0][0]) * (1 + MINIMAL_SALE_MARGIN / 100) and currentPrice > float(close[-2])) or DAILY_VARIABLES[
                'RISKY_SALE_COUNTER'] >= RISKY_SALE_COUNTER_MAX:
                DAILY_VARIABLES['RISKY_SALE_COUNTER'] = 0
                if not sell(quantity):
                    print(time.strftime("%H:%M:%S") + " Sale operation failed - restart initiated.")
                    return False
                else:
                    print(time.strftime("%H:%M:%S") + " Sale initiated. " + "Current price: " + str(currentPrice))
                    return True
            else:
                DAILY_VARIABLES['RISKY_SALE_COUNTER'] += 1
                print(time.strftime("%H:%M:%S") + " Risky sale suggested - counter incremented.")
                return False
        elif inventory and DAILY_VARIABLES['RISKY_SALE_COUNTER'] > 0:
            DAILY_VARIABLES['RISKY_SALE_COUNTER'] = 0
            print(time.strftime("%H:%M:%S") + " Risky sale counter restarted due to inaction.")

        return False


if os.path.isfile("NesModelValid.bin"):
    with open("NesModelValid.bin", "rb") as file:
        model = Model(pickle.load(file))

AGENT = Agent(model=model)


def main(q):
    global BOUGHT, VARIABLES, BACKLOG_IDS, DAILY_VARIABLES, SPECIAL_VARIABLES

    # Get global variables from files
    if os.path.isfile("Bought.json"):
        with open("Bought.json", "r") as file:
            BOUGHT = simplejson.load(file)
    if os.path.isfile("Variables.json"):
        with open("Variables.json", "r") as file:
            VARIABLES = simplejson.load(file)
    if os.path.isfile("Backlog_IDs.json"):
        with open("Backlog_IDs.json", "r") as file:
            BACKLOG_IDS = simplejson.load(file)
    if os.path.isfile("Daily_Variables.json"):
        with open("Daily_Variables.json", "r") as file:
            DAILY_VARIABLES = simplejson.load(file)
    if os.path.isfile("Special_Variables.json"):
        with open("Special_Variables.json", "r") as file:
            SPECIAL_VARIABLES = simplejson.load(file)

    with Client(TOKEN, target=TARGET) as client:
        while True:
            q.put(1)
            # Check if manual exit is triggered or if it's time to stop trading
            if keyboard.is_pressed('esc') or int(time.strftime("%H")) < 10:
                time.sleep(TRY_AGAIN_TIME_SHORT)
                if keyboard.is_pressed('esc') or int(time.strftime("%H")) < 10:
                    if len(BACKLOG_IDS) > 0:
                        for i in range(len(BACKLOG_IDS) - 1, -1, -1):
                            executionReport = getExecutionReport(BACKLOG_IDS[i])
                            if executionReport == EXECUTION_REPORT_STATUS_FILL:
                                reportSale(BACKLOG_IDS[i], i)
                                del BOUGHT[i]
                                del BACKLOG_IDS[i]
                            elif executionReport is None:
                                time.sleep(SLEEP_TIME)
                                continue
                            dumpInfo()
                    reportFinal()
                    q.put("NORMAL_END")
                    return
            # Check trading status
            trading = getTradingStatus()
            if trading == SECURITY_TRADING_STATUS_NORMAL_TRADING:
                # Begin trading
                # Once-per-day Backlog sale set-up
                BACKLOG_IDS = setBacklogForSale()
                if BACKLOG_IDS is None:
                    BACKLOG_IDS = []
                    time.sleep(SLEEP_TIME)
                    continue
                # Check for successful Backlog sales
                if len(BACKLOG_IDS) > 0:
                    for i in range(len(BACKLOG_IDS) - 1, -1, -1):
                        executionReport = getExecutionReport(BACKLOG_IDS[i])
                        if executionReport == EXECUTION_REPORT_STATUS_FILL:
                            reportSale(BACKLOG_IDS[i], i)
                            del BOUGHT[i]
                            del BACKLOG_IDS[i]
                        elif executionReport is None:
                            time.sleep(SLEEP_TIME)
                            continue
                        dumpInfo()
                # Determine the type of the last order
                orderDirection = getOrderDirection()
                if orderDirection is None and not SPECIAL_VARIABLES['FIRST_ORDER']:
                    time.sleep(SLEEP_TIME)
                    continue
                # Check for active orders
                activeOrders = getActiveOrders()
                if activeOrders is None:
                    time.sleep(SLEEP_TIME)
                    continue
                if len(activeOrders) > len(BACKLOG_IDS):
                    if orderDirection == ORDER_DIRECTION_BUY:
                        executionReport = getExecutionReport(VARIABLES['ORDER_ID'])
                        if executionReport != EXECUTION_REPORT_STATUS_FILL and executionReport != EXECUTION_REPORT_STATUS_PARTIALLYFILL:
                            DAILY_VARIABLES['BUY_WAIT_COUNTER'] += 1
                    if DAILY_VARIABLES['BUY_WAIT_COUNTER'] > BUY_WAIT_LIMIT:
                        cancelOrder(VARIABLES['ORDER_ID'])
                        SPECIAL_VARIABLES['FIRST_ORDER'] = True
                        DAILY_VARIABLES['BUY_WAIT_COUNTER'] = 0
                        VARIABLES['ORDER_ID'] = 0
                        print("Buy order cancelled")
                    time.sleep(SLEEP_TIME)
                # Register the successful buy order execution and initiate the sale order
                elif orderDirection == ORDER_DIRECTION_BUY and not SPECIAL_VARIABLES['FIRST_ORDER']:
                    # Check that the last buy order was not yet reported to prevent false reports
                    if len(BOUGHT) <= len(BACKLOG_IDS):
                        price = getOrderPrice()
                        if price is not None:
                            quantity = getOrderQuantity()
                            if quantity is not None:
                                BOUGHT.append([price, quantity])
                                print(time.strftime("%H:%M:%S") + " Registered a purchase. Price:" + str(price) + " Quantity:" + str(quantity) + ".")
                                reportBuy()
                                DAILY_VARIABLES['DAY_LIMIT_COUNTER'] += 1
                                dumpInfo()
                    if len(BOUGHT) > len(BACKLOG_IDS) and DAILY_VARIABLES['DAY_LIMIT_COUNTER'] < 100:
                        AGENT.predict(BOUGHT, BOUGHT[-1][1])
                    time.sleep(SLEEP_TIME)
                # Register the successful sale order and initiate the buy order
                elif SPECIAL_VARIABLES['FIRST_ORDER'] or orderDirection == ORDER_DIRECTION_SELL:
                    # Check that the last sell order was not yet reported to prevent false reports
                    if len(BOUGHT) > len(BACKLOG_IDS):
                        reportSale(VARIABLES['ORDER_ID'])
                        BOUGHT.pop()
                        DAILY_VARIABLES['DAY_LIMIT_COUNTER'] += 1
                        dumpInfo()
                    # Check that the backlog is not too big to prevent exceeding the budget for this bot and check that it's not too late or too early to buy
                    if len(BACKLOG_IDS) < MAX_BACKLOG and 23 > int(time.strftime("%H")) > 10 and DAILY_VARIABLES['DAY_LIMIT_COUNTER'] < 100:
                        if AGENT.predict():
                            if SPECIAL_VARIABLES['FIRST_ORDER']:
                                SPECIAL_VARIABLES['FIRST_ORDER'] = False
                                dumpInfo()
                    time.sleep(SLEEP_TIME)
            elif trading is not None:
                print(time.strftime("%H:%M:%S") + " Instrument is temporary unavailable for trading.")
                for i in range(20):
                    q.put(1)
                    time.sleep(31)
                    q.put(1)
            else:
                print(time.strftime("%H:%M:%S") + " Instrument is unavailable for trading!")
                time.sleep(SLEEP_TIME)


# Get the average price of the share for the last (historyHours) hours
def getAveragePrice():
    with Client(TOKEN, target=TARGET) as client:
        candles = None
        for x in range(0, 2):  # try 2 times
            try:
                candles = client.market_data.get_candles(from_=now() - timedelta(hours=HISTORY_HOURS), to=now(), interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
                                                         instrument_id=INSTRUMENT_ID).candles
            except InvestError as error:
                print("Get average price. Exception in get_candles:", error)
                time.sleep(TRY_AGAIN_TIME_SHORT)  # wait for 1 second before trying again
            else:
                break
        prices = []
        if candles is not None:
            for candle in candles:
                prices.append((quotation_to_decimal(candle.high) + quotation_to_decimal(candle.low)) / 2)
            prices.sort()
            return Decimal(round(prices[int(len(prices) / 2)], VARIABLES['DECIMAL_POINTS']))
    time.sleep(TRY_AGAIN_TIME_LONG)
    return -1


# Give a new sell order
def sell(quantity, old=False, price=0.0):
    with Client(TOKEN, target=TARGET) as client:
        if not old:
            for x in range(0, 3):  # try 3 times
                try:
                    VARIABLES['ORDER_ID'] = client.orders.post_order(instrument_id=INSTRUMENT_ID, account_id=ACCOUNT_ID, direction=ORDER_DIRECTION_SELL,
                                                                     order_type=ORDER_TYPE_MARKET, quantity=quantity).order_id
                except InvestError as error:
                    print("Sell. Exception in post_order (new):", error)
                    out = False
                    time.sleep(TRY_AGAIN_TIME_SHORT)  # wait for 1 second before trying again
                else:
                    out = True
                    dumpInfo()
                    break
        else:
            for x in range(0, 3):  # try 3 times
                try:
                    VARIABLES['ORDER_ID'] = client.orders.post_order(instrument_id=INSTRUMENT_ID, account_id=ACCOUNT_ID, direction=ORDER_DIRECTION_SELL,
                                                                     order_type=ORDER_TYPE_LIMIT, quantity=quantity,
                                                                     price=decimal_to_quotation(round(Decimal(price), VARIABLES['DECIMAL_POINTS']))).order_id
                except InvestError as error:
                    print("Sell. Exception in post_order (old):", error)
                    out = False
                    time.sleep(TRY_AGAIN_TIME_SHORT)  # wait for 1 second before trying again
                else:
                    out = True
                    dumpInfo()
                    break
        return out


# Register the last sell order and output statistics
def reportSale(ID, INDEX=-1):
    with Client(TOKEN, target=TARGET) as client:
        with open("Stats.txt", "a") as file:
            sold = None
            for x in range(0, 3):  # try 3 times
                try:
                    sold = float(quotation_to_decimal(client.orders.get_order_state(account_id=ACCOUNT_ID, order_id=ID).average_position_price))
                except InvestError as error:
                    print("Report sale. Exception in get_order_state or get_instrument_by:", error)
                    time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
                else:
                    break
            if sold is not None:
                if TARGET != INVEST_GRPC_API:
                    sold -= VARIABLES['MIN_PRICE_INCREMENT']
                profitPercent = (sold - float(BOUGHT[INDEX][0])) / float(BOUGHT[INDEX][0]) * 100
                profitUnit = ((sold - float(BOUGHT[INDEX][0])) * BOUGHT[INDEX][1]) * VARIABLES['LOT_SIZE']
                DAILY_VARIABLES['DAILY_PROFIT_PERCENT'] += profitPercent
                DAILY_VARIABLES['DAILY_PROFIT_UNIT'] += profitUnit
                VARIABLES['TOTAL_PROFIT_PERCENT'] += profitPercent
                VARIABLES['TOTAL_PROFIT_UNIT'] += profitUnit
                DAILY_VARIABLES['DAILY_SALES'] += 1
                dumpInfo()
                if ID == VARIABLES['ORDER_ID']:
                    out = time.strftime("%H:%M:%S") + "  Sold " + str(BOUGHT[INDEX][1]) + " share(s) for " + str(sold) + "   Profit " + str(profitPercent) + "%"
                    file.write(out + "\n")
                    print(time.strftime("%H:%M:%S") + " Registered a sale of " + str(BOUGHT[INDEX][1]) + " share(s) for " + str(sold) + "   Profit " + str(profitPercent) + "%" + ".")
                else:
                    out = time.strftime("%H:%M:%S") + "  Sold " + str(BOUGHT[INDEX][1]) + " Backlog share(s) for " + str(sold) + "   Profit " + str(profitPercent) + "%"
                    file.write(out + "\n")
                    print(time.strftime("%H:%M:%S") + " Registered a sale of " + str(BOUGHT[INDEX][1]) + " Backlog share(s) for " + str(sold) + "   Profit " + str(profitPercent) + "%" + ".")
            return


# Give a new buy order
def buy(curPrice):
    with Client(TOKEN, target=TARGET) as client:
        for x in range(0, 3):  # try 3 times
            try:
                VARIABLES['ORDER_ID'] = client.orders.post_order(instrument_id=INSTRUMENT_ID, account_id=ACCOUNT_ID, direction=ORDER_DIRECTION_BUY, order_type=ORDER_TYPE_LIMIT,
                                                                 quantity=getQuantity(curPrice), price=decimal_to_quotation(round(Decimal(curPrice), VARIABLES['DECIMAL_POINTS']))).order_id
            except InvestError as error:
                print("Buy. Exception in post_order:", error)
                out = False
                time.sleep(TRY_AGAIN_TIME_SHORT)  # wait for 1 second before trying again
            else:
                out = True
                dumpInfo()
                break
        return out


# Register the last buy order and output statistics
def reportBuy():
    dumpInfo()
    with open("Stats.txt", "a") as file:
        DAILY_VARIABLES['DAILY_BUYS'] += 1
        out = time.strftime("%H:%M:%S") + "  Bought " + str(BOUGHT[-1][1]) + " stock for " + str(BOUGHT[-1][0])
        file.write(out + "\n")
        dumpInfo()
        return


# Dump permanent info into files
def dumpInfo(final=False):
    if not final:
        with open("Variables.json", "w") as file:
            simplejson.dump(VARIABLES, file, sort_keys=True, indent=4)
        with open("Bought.json", "w") as file:
            simplejson.dump(BOUGHT, file, sort_keys=True, indent=4)
        with open("Backlog_IDs.json", "w") as file:
            simplejson.dump(BACKLOG_IDS, file, sort_keys=True, indent=4)
        with open("Daily_Variables.json", "w") as file:
            simplejson.dump(DAILY_VARIABLES, file, sort_keys=True, indent=4)
        with open("Special_Variables.json", "w") as file:
            simplejson.dump(SPECIAL_VARIABLES, file, sort_keys=True, indent=4)
    else:
        with open("Backlog_IDs.json", "w") as file:
            simplejson.dump(BACKLOG_IDS, file, sort_keys=True, indent=4)
        with open("Daily_Variables.json", "w") as file:
            simplejson.dump(DAILY_VARIABLES, file, sort_keys=True, indent=4)
        with open("Special_Variables.json", "w") as file:
            simplejson.dump(SPECIAL_VARIABLES, file, sort_keys=True, indent=4)


# Final report for the day
def reportFinal():
    with open("Stats.txt", "a") as file:
        out = "\n" + time.strftime("%H:%M:%S") + " Day's end report:" + "\n\nBacklog at day's start:" + str(
            DAILY_VARIABLES['DAY_START_BACKLOG']) + "\n" + "Backlog at day's end:" + str(len(BACKLOG_IDS)) + "\n" + "\n\nBought stock:" + str(
            DAILY_VARIABLES['DAILY_BUYS']) + " times\nSold stock:" + str(DAILY_VARIABLES['DAILY_SALES']) + " times\nDaily profit in percent:" + str(
            DAILY_VARIABLES['DAILY_PROFIT_PERCENT']) + "%\nProfit per sale:" + str(
            round(DAILY_VARIABLES['DAILY_PROFIT_PERCENT'] / (DAILY_VARIABLES['DAILY_SALES'] + 1e-7), 3)) + "\nDaily profit in units:" + str(
            DAILY_VARIABLES['DAILY_PROFIT_UNIT']) + "\nTotal profit in percent:" + str(VARIABLES['TOTAL_PROFIT_PERCENT']) + "%\nTotal profit in units (lot size accounted):" + str(
            VARIABLES['TOTAL_PROFIT_UNIT'])
        file.write(out)
    newStatsFileName = "Old Reports/Rep" + VARIABLES['DATE'] + ".txt"
    if not os.path.isdir("Old Reports"):
        os.mkdir("Old Reports")
    os.rename("Stats.txt", newStatsFileName)
    VARIABLES['MONTHLY_PROFIT_PERCENT'] += DAILY_VARIABLES['DAILY_PROFIT_PERCENT']
    VARIABLES['MONTHLY_PROFIT_UNIT'] += DAILY_VARIABLES['DAILY_PROFIT_UNIT']
    dumpInfo()


# Final report for the month
def reportMonthly():
    monthFile = "Old Reports/Monthly Report " + VARIABLES['MONTH'] + ".txt"
    if not os.path.isdir("Old Reports"):
        os.mkdir("Old Reports")
    with open(monthFile, "w") as file:
        out = "Month's end report:" + "\nMonthly profit in percent:" + str(VARIABLES['MONTHLY_PROFIT_PERCENT']) + "%\nMonthly profit in units (lot size accounted):" + str(
            VARIABLES['MONTHLY_PROFIT_UNIT'] * VARIABLES['LOT_SIZE'])
        file.write(out)
    VARIABLES['MONTHLY_PROFIT_PERCENT'] = 0
    VARIABLES['MONTHLY_PROFIT_UNIT'] = 0
    dumpInfo()


# Determine the quantity of lots traded based on TOTAL_PROFIT_UNIT (VARIABLES['TOTAL_PROFIT_UNIT'])
def getQuantity(curPrice):
    while True:
        VARIABLES['ALLOWED_ORDERS_WITH_NEW_QUANTITY'] = math.floor((VARIABLES['TOTAL_PROFIT_UNIT'] * 0.8) / (float(curPrice) * VARIABLES['LOT_SIZE'])) - (
                MAX_BACKLOG * (VARIABLES['OLD_QUANTITY'] - INITIAL_QUANTITY))
        if VARIABLES['ALLOWED_ORDERS_WITH_NEW_QUANTITY'] >= MAX_BACKLOG:
            VARIABLES['OLD_QUANTITY'] = VARIABLES['NEW_QUANTITY']
            VARIABLES['NEW_QUANTITY'] += 1
            dumpInfo()
        else:
            VARIABLES['ALLOWED_ORDERS_WITH_NEW_QUANTITY'] = 0
            dumpInfo()
            break
    return VARIABLES['OLD_QUANTITY']


def getDecimalPoints():
    with Client(TOKEN, target=TARGET) as client:
        for x in range(0, 3):  # try 3 times
            try:
                min_price_increment = client.instruments.get_instrument_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_UID, id=INSTRUMENT_ID).instrument.min_price_increment
                VARIABLES['DECIMAL_POINTS'] = 9 - len(str(min_price_increment.nano)) + 1
                VARIABLES['MIN_PRICE_INCREMENT'] = float(quotation_to_decimal(min_price_increment))
            except InvestError as error:
                print("Get decimal points. Exception in get_instrument_by:", error)
                out = False
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                out = True
                break
        return out


def getLotSize():
    with Client(TOKEN, target=TARGET) as client:
        for x in range(0, 3):  # try 3 times
            try:
                lotSize = client.instruments.get_instrument_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_UID, id=INSTRUMENT_ID).instrument.lot
                VARIABLES['LOT_SIZE'] = lotSize
            except InvestError as error:
                print("Get lot size. Exception in get_instrument_by:", error)
                out = False
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                out = True
                break
        return out


def getActiveOrders():
    with Client(TOKEN, target=TARGET) as client:
        activeOrders = None
        for x in range(0, 3):  # try 3 times
            try:
                activeOrders = client.orders.get_orders(account_id=ACCOUNT_ID).orders
            except InvestError as error:
                print("Check for active orders. Exception in get_orders:", error)
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                break
        return activeOrders


def getOrderDirection():
    with Client(TOKEN, target=TARGET) as client:
        orderDirection = None
        if not SPECIAL_VARIABLES['FIRST_ORDER']:
            for x in range(0, 3):  # try 3 times
                try:
                    orderDirection = client.orders.get_order_state(account_id=ACCOUNT_ID, order_id=VARIABLES['ORDER_ID']).direction
                except InvestError as error:
                    print("Order type. Exception in get_order_state:", error)
                    time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
                else:
                    break
        return orderDirection


def setBacklogForSale():
    global AGENT
    if not SPECIAL_VARIABLES['BACKLOG_SET_FOR_SALE']:
        # Get the allowed decimal point for the price of the lot
        if not getDecimalPoints():
            time.sleep(SLEEP_TIME)
            return None
        # Get the lot size
        if not getLotSize():
            time.sleep(SLEEP_TIME)
            return None
        # Delete any previous orders
        for i in getActiveOrders():
            cancelOrder(i.order_id)
        # Get backlog at day start
        DAILY_VARIABLES['DAY_START_BACKLOG'] = len(BOUGHT)
        # Check for monthly report and get month and date at day start
        if VARIABLES['MONTH'] != "" and time.strftime("%m") != VARIABLES['MONTH']:
            reportMonthly()
        VARIABLES['DATE'] = time.strftime("%d-%m-%y")
        VARIABLES['MONTH'] = time.strftime("%m")
        # Update the model with the previous day's data
        if os.path.isfile("NesModelValid.bin"):
            with open("NesModelValid.bin", "rb") as file:
                model = Model(pickle.load(file))

        AGENT = Agent(model=model)
        # Deal with the Backlog
        """temp_id = VARIABLES['ORDER_ID']
        for purchase in BOUGHT:
            if not sell(purchase[1], True, float(purchase[0]) * (BACKLOG_SELLING_MARGIN / 100 + 1)):
                if len(BACKLOG_IDS) == 0:
                    time.sleep(SLEEP_TIME)
                    return None
                else:
                    while not sell(purchase[1], True, purchase[0] * (BACKLOG_SELLING_MARGIN / 100 + 1)):
                        if keyboard.is_pressed('esc') or int(time.strftime("%H")) < 10:
                            return None
                        time.sleep(SLEEP_TIME)
            print(time.strftime("%H:%M:%S") + " Backlog asset sale initiated for " + str(float(purchase[0]) * (BACKLOG_SELLING_MARGIN / 100 + 1)) + ".")
            BACKLOG_IDS.append(VARIABLES['ORDER_ID'])
        VARIABLES['ORDER_ID'] = temp_id"""
        SPECIAL_VARIABLES['BACKLOG_SET_FOR_SALE'] = True
        dumpInfo()
    return BACKLOG_IDS


def getTradingStatus():
    with Client(TOKEN, target=TARGET) as client:
        trading = None
        for x in range(0, 3):  # try 3 times
            try:
                trading = client.market_data.get_trading_status(instrument_id=INSTRUMENT_ID).trading_status
            except InvestError as error:
                print("Check trading status. Exception in get_trading_status:", error)
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                break
        return trading


def getExecutionReport(i):
    with Client(TOKEN, target=TARGET) as client:
        executionReport = None
        for x in range(0, 3):  # try 3 times
            try:
                executionReport = client.orders.get_order_state(account_id=ACCOUNT_ID, order_id=i).execution_report_status
            except InvestError as error:
                print("Check Backlog for sales. Exception in get_order_state:", error)
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                break
        return executionReport


def getOrderPrice():
    with Client(TOKEN, target=TARGET) as client:
        price = None
        for x in range(0, 3):  # try 3 times
            try:
                price = quotation_to_decimal(client.orders.get_order_state(account_id=ACCOUNT_ID, order_id=str(VARIABLES['ORDER_ID'])).average_position_price)
            except InvestError as error:
                print("Purchase registration - price. Exception in get_order_state:", error)
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                break
        return price


def getOrderQuantity():
    with Client(TOKEN, target=TARGET) as client:
        quantity = None
        for x in range(0, 3):  # try 3 times
            try:
                quantity = client.orders.get_order_state(account_id=ACCOUNT_ID, order_id=str(VARIABLES['ORDER_ID'])).lots_executed
            except InvestError as error:
                print("Purchase registration - quantity. Exception in get_order_state:", error)
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                break
        return quantity


def moveToBacklog():
    """if not sell(BOUGHT[-1][1], True, float(BOUGHT[-1][0]) * (BACKLOG_SELLING_MARGIN / 100 + 1)):
        time.sleep(SLEEP_TIME)
        return False
    print(time.strftime("%H:%M:%S") + " Current order moved to Backlog, it's sale is initiated for " + str(float(BOUGHT[-1][0]) * (BACKLOG_SELLING_MARGIN / 100 + 1)) + ".")
    BACKLOG_IDS.append(VARIABLES['ORDER_ID'])
    VARIABLES['ORDER_ID'] = 0
    SPECIAL_VARIABLES['FIRST_ORDER'] = True
    dumpInfo()
    with open("Stats.txt", "a") as file:
        out = time.strftime("%H:%M:%S") + "  Current order moved to Backlog."
        file.write(out + "\n")"""
    return True


def cancelOrder(order):
    with Client(TOKEN, target=TARGET) as client:
        canceled = None
        for x in range(0, 3):  # try 3 times
            try:
                canceled = client.orders.cancel_order(account_id=ACCOUNT_ID, order_id=order)
            except InvestError as error:
                print("Order cancellation. Exception in cancel_order:", error)
                time.sleep(TRY_AGAIN_TIME_LONG)  # wait for 30 seconds before trying again
            else:
                break
        return canceled


def getCurrentPrice(maximalPrice=None):
    with Client(TOKEN, target=TARGET) as client:
        currentPrice = None
        for x in range(0, 2):  # try 2 times
            try:
                currentPrice = quotation_to_decimal(client.market_data.get_last_prices(instrument_id=[INSTRUMENT_ID]).last_prices[0].price)
            except InvestError as error:
                print("Exception in get_last_prices:", error)
                if maximalPrice is not None:
                    currentPrice = maximalPrice
                time.sleep(TRY_AGAIN_TIME_SHORT)  # wait for 1 second before trying again
            else:
                break
        if currentPrice is None:
            time.sleep(TRY_AGAIN_TIME_LONG)
        return currentPrice


def pinger(q):
    while True:
        try:
            if q.get(timeout=100.0) == "NORMAL_END":
                q.put("NORMAL_END")
        except queue.Empty:
            q.put("EMERGENCY_END")


def mainQueue():
    q = multiprocessing.Queue()
    mainProcess = multiprocessing.Process(target=main, args=(q,))
    pingerProcess = multiprocessing.Process(target=pinger, args=(q,))
    pingerProcess.daemon = True
    mainProcess.start()
    pingerProcess.start()
    while True:
        msg = q.get()
        if msg == "EMERGENCY_END" or msg == "NORMAL_END":
            mainProcess.terminate()
            time.sleep(SLEEP_TIME)
            if not mainProcess.is_alive():
                mainProcess.join()
                q.close()
                break
    if msg == "EMERGENCY_END":
        print(time.strftime("%H:%M:%S") + " Program not responding - restart initiated.")
        return False
    return True


if __name__ == "__main__":
    # Set-up key listener to exit the program at will
    while True:
        if keyboard.is_pressed('esc'):
            time.sleep(TRY_AGAIN_TIME_SHORT)
            if keyboard.is_pressed('esc'):
                exit()
        if 9 < int(time.strftime("%H")) and time.strftime("%a") != "Sun" and time.strftime("%a") != "Sat":
            print(time.strftime("%H:%M:%S") + " Trading started.")
            if not mainQueue():
                continue
            print(time.strftime("%H:%M:%S") + " Trading concluded.")
            for key, value in DAILY_VARIABLES.items():
                DAILY_VARIABLES[key] = 0
            SPECIAL_VARIABLES['BACKLOG_SET_FOR_SALE'] = False
            # SPECIAL_VARIABLES['FIRST_ORDER'] = True
            BACKLOG_IDS.clear()
            dumpInfo(final=True)
        else:
            time.sleep(SLEEP_TIME)
        """if int(time.strftime("%H")) == 0:
            time.sleep(120)
            subprocess.call(['python', 'formatData.py'])
            time.sleep(120)
            subprocess.call(['python', 'NEStrTCSG.py'])
            time.sleep(3600)"""

