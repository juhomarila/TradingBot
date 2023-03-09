import requests
import datetime
import time
import os
import csv
from alpha_vantage.timeseries import TimeSeries
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from yahoofinancials import YahooFinancials

from .machinelearning import train_machine_learning_model_future_values, train_machine_learning_model
from .models import signals, finnish_stock_daily
from .serializers import finnish_stock_daily_serializer, signals_serializer
from .indicators import calculate_adl, calculate_adx, calculate_obv, calculate_rsi, \
    calculate_aroon, calculate_macd, calculate_BBP8, calculate_sd, find_optimum_buy_sell_points, calculate_ema


def process_csv_data(request):
    print(os.getcwd())
    folder_path = '/home/jmarila/PycharmProjects/marketBot/marketBot/charts'
    symbol_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            symbol = filename.split("-")[0]
            symbol_list.append(symbol)
            # with open(os.path.join(folder_path, filename), newline='', encoding='utf-8') as csvfile:
            #     reader = csv.reader(csvfile, delimiter=';')
            #     counter = 0
            #     for row in reader:
            #         if counter < 2:
            #             counter += 1
            #             continue
            #         if (len(row[0]) == 0 or len(row[1]) == 0 or len(row[2]) == 0 or len(row[3]) == 0
            #                 or len(row[4]) == 0 or len(row[5]) == 0 or len(row[6]) == 0 or len(row[7]) == 0
            #                 or len(row[8]) == 0 or len(row[9]) == 0 or len(row[10]) == 0):
            #             continue
            #         else:
            #             date = datetime.datetime.strptime(row[0], '%Y-%m-%d').date()
            #             bid = float(row[1].replace(",", "."))
            #             ask = float(row[2].replace(",", "."))
            #             open_price = float(row[3].replace(",", "."))
            #             high_price = float(row[4].replace(",", "."))
            #             low_price = float(row[5].replace(",", "."))
            #             closing_price = float(row[6].replace(",", "."))
            #             average_price = float(row[7].replace(",", "."))
            #             total_volume = int(float(row[8].replace(",", ".")))
            #             turnover = float(row[9].replace(",", "."))
            #             trades = int(float(row[10].replace(",", ".")))
            #
            #             print(symbol, date, bid, ask, open_price, high_price, low_price,
            #                   closing_price, average_price, total_volume, turnover, trades)
            #             if not finnish_stock_daily.objects.filter(symbol=symbol, date=date).exists():
            #                 finnish_stock_daily.objects.create(symbol=symbol, date=date, bid=bid, ask=ask,
            #                                                    open=open_price, high=high_price,
            #                                                    low=low_price, close=closing_price,
            #                                                    average=average_price, volume=total_volume,
            #                                                    turnover=turnover, trades=trades)
    # train_machine_learning_model()
    # train_machine_learning_model_future_values()
    for i in range(len(symbol_list)):
        calculate_BBP8(symbol_list[i], finnish_stock_daily, True, 5)
        calculate_sd(symbol_list[i], finnish_stock_daily, True, 5)
        calculate_rsi(symbol_list[i], finnish_stock_daily, True, 5)
        calculate_rsi(symbol_list[i], finnish_stock_daily, True, 7)
        calculate_macd(symbol_list[i], finnish_stock_daily, True, 5)
        calculate_obv(symbol_list[i], finnish_stock_daily, True)
        calculate_adl(symbol_list[i], finnish_stock_daily, True)
        calculate_adx(symbol_list[i], finnish_stock_daily, True, 5)
        calculate_ema(symbol_list[i], finnish_stock_daily, True, 5)
    #     calculate_BBP8(symbol_list[i], finnish_stock_daily, false, 8)
    #     calculate_aroon(symbol_list[i], finnish_stock_daily, 14)
    #     calculate_ema(symbol_list[i], finnish_stock_daily, 20)
    #     calculate_ema(symbol_list[i], finnish_stock_daily, 50)
    #     calculate_ema(symbol_list[i], finnish_stock_daily, 100)
    #     calculate_ema(symbol_list[i], finnish_stock_daily, 200)
    #     calculate_macd(symbol_list[i], finnish_stock_daily, 9)
    #     calculate_rsi(symbol_list[i], finnish_stock_daily, 7)
    #     calculate_rsi(symbol_list[i], finnish_stock_daily, 14)
    #     calculate_rsi(symbol_list[i], finnish_stock_daily, False, 50)
    #     calculate_obv(symbol_list[i], finnish_stock_daily)
    #     calculate_adl(symbol_list[i], finnish_stock_daily)
    #     calculate_adx(symbol_list[i], finnish_stock_daily, 14)
    #     calculate_sd(symbol_list[i], finnish_stock_daily, 14)
    return HttpResponse(status=201)


def find_buy_sell_points(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
    for i in range(len(symbol_list)):
        print(symbol_list[i]['symbol'])
        find_optimum_buy_sell_points(symbol_list[i]['symbol'], True)
    return HttpResponse(status=201)


def process_daily_data(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
    for i in range(len(symbol_list)):
        calculate_adx(symbol_list[i]['symbol'], finnish_stock_daily, True)
        calculate_aroon(symbol_list[i]['symbol'], finnish_stock_daily, True)
        calculate_macd(symbol_list[i]['symbol'], finnish_stock_daily, True)
        calculate_rsi(symbol_list[i]['symbol'], finnish_stock_daily, True)
    return HttpResponse(status=201)


def get_daily_data(request):
    names = finnish_stock_daily.objects.values('symbol').distinct()
    today = datetime.date.today()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    for i in range(len(names)):
        if names[i]['symbol'] == 'NDA':
            data = YahooFinancials(names[i]['symbol'] + '-FI.HE').get_historical_price_data(start_date=str(today),
                                                                                            end_date=str(tomorrow),
                                                                                            time_interval='daily')
            stock_data = data[names[i]['symbol'] + '-FI.HE']['prices'][0]
            if not finnish_stock_daily.objects.filter(symbol=names[i]['symbol'],
                                                      date=stock_data['formatted_date']).exists():
                finnish_stock_daily.objects.create(symbol=names[i]['symbol'], date=stock_data['formatted_date'],
                                                   open=stock_data['open'], high=stock_data['high'],
                                                   low=stock_data['low'], close=stock_data['close'],
                                                   volume=stock_data['volume'])
        else:
            data = YahooFinancials(names[i]['symbol'] + '.HE').get_historical_price_data(start_date=str(today),
                                                                                         end_date=str(tomorrow),
                                                                                         time_interval='daily')
            stock_data = data[names[i]['symbol'] + '.HE']['prices'][0]
            if not finnish_stock_daily.objects.filter(symbol=names[i]['symbol'],
                                                      date=stock_data['formatted_date']).exists():
                finnish_stock_daily.objects.create(symbol=names[i]['symbol'], date=stock_data['formatted_date'],
                                                   open=stock_data['open'], high=stock_data['high'],
                                                   low=stock_data['low'], close=stock_data['close'],
                                                   volume=stock_data['volume'])
    return HttpResponse(status=201)


def index(request):
    q = request.GET.get('q', None)
    if q:
        return HttpResponse('Haista vittu')
    else:
        return HttpResponse("Hello, world. You're at the polls index.")
