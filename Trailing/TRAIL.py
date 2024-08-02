from datetime import timedelta
import time
import simplejson
from decimal import *
import os
import math
import keyboard
import multiprocessing
import queue

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
HISTORY_HOURS = 3.5
SLEEP_TIME = 2
TRY_AGAIN_TIME_SHORT = 2
TRY_AGAIN_TIME_LONG = 60
SELLING_MARGIN = 0.3
BUYING_MARGIN = 0.9
ALLOWED_TRAIL_SELL_PERCENT = 35
ALLOWED_TRAIL_BUY_PERCENT = 15
BACKLOG_SELLING_MARGIN = 0.05
STRICT_SELLING_MARGIN = 0.2
STRICT_BUYING_MARGIN = 0

STARTING_MONEY = 9000

DROP_OFF_START = 0.65
DROP_OFF_LIMIT = 10
DROP_OFF_TIME = 11 * (60 / SLEEP_TIME)

BOUGHT = []
VARIABLES = {'ORDER_ID': 0, 'TOTAL_PROFIT_PERCENT': 0, 'TOTAL_PROFIT_UNIT': 0, 'DECIMAL_POINTS': 2, 'BACKLOG_SET_FOR_SALE': False, 'FIRST_ORDER': True, 'DATE': "", 'LOT_SIZE': 0,
             'MONTHLY_PROFIT_PERCENT': 0, 'MONTHLY_PROFIT_UNIT': 0, 'MONTH': "", "DROP_OFF_NUM": 0, 'MIN_PRICE_INCREMENT': 0}
BACKLOG_IDS = []
DAILY_VARIABLES = {'DAILY_PROFIT_PERCENT': 0, 'DAILY_PROFIT_UNIT': 0, 'DAILY_BUYS': 0, 'DAILY_SALES': 0, 'DAY_START_BACKLOG': 0}


def main(q):
    global BOUGHT, VARIABLES, BACKLOG_IDS, DAILY_VARIABLES

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
                    for key, value in DAILY_VARIABLES.items():
                        DAILY_VARIABLES[key] = 0
                    VARIABLES['BACKLOG_SET_FOR_SALE'] = False
                    VARIABLES['FIRST_ORDER'] = True
                    BACKLOG_IDS.clear()
                    dumpInfo()
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
                # Check for successful Backlog sales and for Drop Offs
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
                    for i in range(len(BACKLOG_IDS) - 1, -1, -1):
                        backlogDropOff = checkBacklogDropOff(i)
                        if backlogDropOff:
                            cancelOrder(BACKLOG_IDS[i])
                            del BACKLOG_IDS[i]
                        elif backlogDropOff is None:
                            time.sleep(SLEEP_TIME)
                            continue
                        dumpInfo()
                # Determine the type of the last order
                orderDirection = getOrderDirection()
                if orderDirection is None and not VARIABLES['FIRST_ORDER']:
                    time.sleep(SLEEP_TIME)
                    continue
                # Check for active orders
                activeOrders = getActiveOrders()
                if activeOrders is None:
                    time.sleep(SLEEP_TIME)
                    continue
                if len(activeOrders) > len(BACKLOG_IDS):
                    time.sleep(SLEEP_TIME)
                # Register the successful buy order execution and initiate the sale order
                elif orderDirection == ORDER_DIRECTION_BUY:
                    # Check that the last buy order was not yet reported to prevent false reports
                    if len(BOUGHT) <= len(BACKLOG_IDS):
                        price = getOrderPrice()
                        if price is not None:
                            quantity = getOrderQuantity()
                            if quantity is not None:
                                BOUGHT.append([price, quantity])
                                print(time.strftime("%H:%M:%S") + " Registered a purchase. Price:" + str(price) + " Quantity:" + str(quantity) + ".")
                                reportBuy()
                    if len(BOUGHT) > len(BACKLOG_IDS):
                        trailToSell(BOUGHT[-1][1], q)
                    time.sleep(SLEEP_TIME)
                # Register the successful sale order and initiate the buy order
                elif VARIABLES['FIRST_ORDER'] or orderDirection == ORDER_DIRECTION_SELL:
                    # Check that the last sell order was not yet reported to prevent false reports
                    if len(BOUGHT) > len(BACKLOG_IDS):
                        reportSale(VARIABLES['ORDER_ID'])
                        BOUGHT.pop()
                        dumpInfo()
                    # Check that the backlog is not too big to prevent exceeding the budget for this bot and check that it's not too late or too early to buy
                    if len(BACKLOG_IDS) < MAX_BACKLOG and 23 > int(time.strftime("%H")) > 10:
                        if trailToBuy(q):
                            if VARIABLES['FIRST_ORDER']:
                                VARIABLES['FIRST_ORDER'] = False
                                dumpInfo()
                    time.sleep(SLEEP_TIME)
            elif trading is not None:
                print(time.strftime("%H:%M:%S") + " Instrument is temporary unavailable for trading.")
                q.put(1)
                time.sleep(300)
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
                    print(round(Decimal(price), VARIABLES['DECIMAL_POINTS']))
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
                    print(
                        time.strftime("%H:%M:%S") + " Registered a sale of " + str(BOUGHT[INDEX][1]) + " share(s) for " + str(sold) + "   Profit " + str(profitPercent) + "%" + ".")
                else:
                    out = time.strftime("%H:%M:%S") + "  Sold " + str(BOUGHT[INDEX][1]) + " Backlog share(s) for " + str(sold) + "   Profit " + str(profitPercent) + "%"
                    file.write(out + "\n")
                    print(time.strftime("%H:%M:%S") + " Registered a sale of " + str(BOUGHT[INDEX][1]) + " Backlog share(s) for " + str(sold) + "   Profit " + str(
                        profitPercent) + "%" + ".")
            return


# Give a new buy order
def buy(curPrice):
    with Client(TOKEN, target=TARGET) as client:
        for x in range(0, 3):  # try 3 times
            try:
                VARIABLES['ORDER_ID'] = client.orders.post_order(instrument_id=INSTRUMENT_ID, account_id=ACCOUNT_ID, direction=ORDER_DIRECTION_BUY, order_type=ORDER_TYPE_MARKET,
                                                                 quantity=getQuantity(curPrice)).order_id
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
def dumpInfo():
    with open("Variables.json", "w") as file:
        simplejson.dump(VARIABLES, file, sort_keys=True, indent=4)
    with open("Bought.json", "w") as file:
        simplejson.dump(BOUGHT, file, sort_keys=True, indent=4)
    with open("Backlog_IDs.json", "w") as file:
        simplejson.dump(BACKLOG_IDS, file, sort_keys=True, indent=4)
    with open("Daily_Variables.json", "w") as file:
        simplejson.dump(DAILY_VARIABLES, file, sort_keys=True, indent=4)


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
    return math.floor((STARTING_MONEY + (VARIABLES['TOTAL_PROFIT_UNIT'] * 0.9)) / (float(curPrice) * VARIABLES['LOT_SIZE']))


def trailToBuy(q):
    with Client(TOKEN, target=TARGET) as client:
        print(time.strftime("%H:%M:%S") + " Purchase trailing initiated.")
        while True:
            while True:
                q.put(1)
                # Check if manual exit is triggered or if it's time to stop trading
                if keyboard.is_pressed('esc') or int(time.strftime("%H")) > 22:
                    return False
                startingPrice = getAveragePrice()
                TEMP_BUYING_MARGIN = BUYING_MARGIN
                if startingPrice != -1:
                    currentPrice = getCurrentPrice()
                    if currentPrice is not None:
                        if (startingPrice - currentPrice) / startingPrice * 100 > TEMP_BUYING_MARGIN:
                            minimalPrice = currentPrice
                            print(time.strftime("%H:%M:%S") + " Purchase trailing first cycle concluded. Starting price:" + str(startingPrice) + " Minimal price:" + str(
                                minimalPrice))
                            break
                time.sleep(SLEEP_TIME)
            while True:
                q.put(1)
                # Check if manual exit is triggered or if it's time to stop trading
                if keyboard.is_pressed('esc') or int(time.strftime("%H")) > 22:
                    return False
                currentPrice = getCurrentPrice()
                if currentPrice is not None:
                    if currentPrice <= minimalPrice:
                        if currentPrice < minimalPrice:
                            minimalPrice = currentPrice
                            print(time.strftime("%H:%M:%S") + " Purchase trailing update. Minimal price:" + str(minimalPrice))
                    elif (((startingPrice - minimalPrice) - (startingPrice - currentPrice)) / (startingPrice - minimalPrice) * 100) > ALLOWED_TRAIL_BUY_PERCENT:
                        if currentPrice <= startingPrice * Decimal(1 - STRICT_BUYING_MARGIN / 100):
                            if not buy(currentPrice):
                                print(time.strftime("%H:%M:%S") + " Purchasing operation failed - restart initiated.")
                                break
                            else:
                                print(time.strftime("%H:%M:%S") + " Purchase trailing concluded. Purchase initiated.")
                                return True
                        else:
                            print(time.strftime("%H:%M:%S") + " Purchase trailing second cycle failed - restart initiated.")
                            break
                time.sleep(SLEEP_TIME)


def trailToSell(quantity, q):
    with Client(TOKEN, target=TARGET) as client:
        startingPrice = float(BOUGHT[-1][0])
        print(time.strftime("%H:%M:%S") + " Initiated sale trailing. Starting price:" + str(startingPrice) + " Trigger price:" + str(
            float(startingPrice) + float(startingPrice) * (SELLING_MARGIN / 100)) + ".")
        START = time.time()
        while True:
            while True:
                q.put(1)
                # Check if manual exit is triggered or if it's time to stop trading
                if keyboard.is_pressed('esc') or int(time.strftime("%H")) < 10:
                    return
                END = time.time()
                if (END - START >= 3600) and (len(BACKLOG_IDS) < MAX_BACKLOG - 1) and (startingPrice > float(getAveragePrice()) * (1 + SELLING_MARGIN / 100)):
                    if moveToBacklog():
                        return
                currentPrice = getCurrentPrice()
                if currentPrice is not None:
                    currentPrice = float(currentPrice) - VARIABLES['MIN_PRICE_INCREMENT']
                    if (currentPrice - startingPrice) / startingPrice * 100 >= SELLING_MARGIN:
                        maximalPrice = currentPrice
                        print(time.strftime("%H:%M:%S") + " Sale trailing first cycle concluded. Starting price:" + str(startingPrice) + " Maximal price:" + str(maximalPrice))
                        break
                    else:
                        cur_loss = (currentPrice - startingPrice) / currentPrice * -100
                        if DROP_OFF_LIMIT >= cur_loss >= DROP_OFF_START:
                            if VARIABLES['DROP_OFF_NUM'] >= DROP_OFF_TIME:
                                if not sell(quantity):
                                    print(time.strftime("%H:%M:%S") + " Drop Off operation failed - restart initiated.")
                                    break
                                else:
                                    print(time.strftime("%H:%M:%S") + " Drop Off operation concluded. Sale initiated.")
                                    VARIABLES['DROP_OFF_NUM'] = 0
                                    return
                            else:
                                VARIABLES['DROP_OFF_NUM'] += 1
                time.sleep(SLEEP_TIME)
            while True:
                q.put(1)
                # Check if manual exit is triggered or if it's time to stop trading
                if keyboard.is_pressed('esc') or int(time.strftime("%H")) < 10:
                    return
                currentPrice = getCurrentPrice(maximalPrice)
                if currentPrice is not None:
                    currentPrice = float(currentPrice) - VARIABLES['MIN_PRICE_INCREMENT']
                    if currentPrice >= maximalPrice:
                        if currentPrice > maximalPrice:
                            maximalPrice = currentPrice
                            print(time.strftime("%H:%M:%S") + " Sale trailing update. Maximal price:" + str(maximalPrice))
                    elif (((maximalPrice - startingPrice) - (currentPrice - startingPrice)) / (maximalPrice - startingPrice) * 100) > ALLOWED_TRAIL_SELL_PERCENT:
                        if currentPrice >= startingPrice * (STRICT_SELLING_MARGIN / 100 + 1):
                            if not sell(quantity):
                                print(time.strftime("%H:%M:%S") + " Sale operation failed - restart initiated.")
                                break
                            else:
                                print(time.strftime("%H:%M:%S") + " Sale trailing concluded. Sale initiated.")
                                return
                        else:
                            print(time.strftime("%H:%M:%S") + " Sale trailing second cycle failed - restart initiated.")
                            break
                time.sleep(SLEEP_TIME)


def getDecimalPoints():
    with Client(TOKEN, target=TARGET) as client:
        for x in range(0, 3):  # try 3 times
            try:
                min_price_increment = client.instruments.get_instrument_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_UID, id=INSTRUMENT_ID).instrument.min_price_increment
                VARIABLES['MIN_PRICE_INCREMENT'] = float(quotation_to_decimal(min_price_increment))
                VARIABLES['DECIMAL_POINTS'] = 9 - len(str(min_price_increment.nano)) + 1
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
        if not VARIABLES['FIRST_ORDER']:
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
    if not VARIABLES['BACKLOG_SET_FOR_SALE']:
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
        # Deal with the Backlog
        temp_id = VARIABLES['ORDER_ID']
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
        VARIABLES['ORDER_ID'] = temp_id
        VARIABLES['BACKLOG_SET_FOR_SALE'] = True
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


def getExecutionReport(id):
    with Client(TOKEN, target=TARGET) as client:
        executionReport = None
        for x in range(0, 3):  # try 3 times
            try:
                executionReport = client.orders.get_order_state(account_id=ACCOUNT_ID, order_id=id).execution_report_status
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
    if not sell(BOUGHT[-1][1], True, float(BOUGHT[-1][0]) * (BACKLOG_SELLING_MARGIN / 100 + 1)):
        time.sleep(SLEEP_TIME)
        return False
    print(time.strftime("%H:%M:%S") + " Current order moved to Backlog, it's sale is initiated for " + str(float(BOUGHT[-1][0]) * (BACKLOG_SELLING_MARGIN / 100 + 1)) + ".")
    BACKLOG_IDS.append(VARIABLES['ORDER_ID'])
    VARIABLES['ORDER_ID'] = 0
    VARIABLES['FIRST_ORDER'] = True
    dumpInfo()
    with open("Stats.txt", "a") as file:
        out = time.strftime("%H:%M:%S") + "  Current order moved to Backlog."
        file.write(out + "\n")
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


def checkBacklogDropOff(i):
    startingPrice = float(BOUGHT[i][0])
    currentPrice = getCurrentPrice()
    if currentPrice is not None:
        currentPrice = float(currentPrice) - VARIABLES['MIN_PRICE_INCREMENT']
        cur_loss = (currentPrice - startingPrice) / currentPrice * -100
        if DROP_OFF_LIMIT >= cur_loss >= DROP_OFF_START:
            if VARIABLES['DROP_OFF_NUM'] >= DROP_OFF_TIME:
                if not sell(BOUGHT[i][1]):
                    print(time.strftime("%H:%M:%S") + " Backlog Drop Off operation failed.")
                else:
                    print(time.strftime("%H:%M:%S") + " Backlog Drop Off operation concluded. Sale initiated.")
                    VARIABLES['DROP_OFF_NUM'] = 0
                    return True
            else:
                VARIABLES['DROP_OFF_NUM'] += 1
    else:
        return currentPrice
    return False


def pinger(q):
    while True:
        try:
            if q.get(timeout=400.0) == "NORMAL_END":
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
        else:
            time.sleep(SLEEP_TIME)
