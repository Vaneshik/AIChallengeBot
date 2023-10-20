from telebot import TeleBot, types
from dotenv import load_dotenv
import os

import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from inference import my_features, read_signal, clean_signal, predict

load_dotenv()
bot_token = os.getenv("BOT_TOKEN")
signal = None
meta_features = None
bot = TeleBot(bot_token)

HELP_MESSAGE = """
Команды:
/start или /help - вывести данное сообщение 🤓
/uploadECG - загрузить данные 📁 
/plot - построить график сигнала 📊
/genFeatures - сгенерировать признаки по сигналу 🧐
/predict - предсказать диагноз 😎
"""


@bot.message_handler(commands=["start", "help"])
def send_welcome(message: types.Message):
    bot.reply_to(message, HELP_MESSAGE)


@bot.message_handler(commands=["uploadECG"])
def send_welcome(message):
    msg = bot.reply_to(message, "Пожалуйста пришлите файл в формате .npy")
    bot.register_next_step_handler(msg, process_ecg_upload)


def process_ecg_upload(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("new_file.npy", "wb") as new_file:
            new_file.write(downloaded_file)

        message_text = """Файл загружен! Теперь пришлите метаданные в формате
'<age> <sex> <height> <weight>'

Где:
* age - возраст пациента в годах;
* sex - пол пациента, 0 - мужчина / 1 - женщина;
* height - рост пациента в см, дробная часть отделяется точкой;
* age - вес пациента в кг, дробная часть отделяется точкой.

Если какое-то поле отсутствует, то поставьте прочерк '-'"""

        msg = bot.reply_to(message, message_text)
        bot.register_next_step_handler(msg, process_ecg_meta)
    except Exception as e:
        print(e)
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")


def process_ecg_meta(message):
    global signal, meta
    try:
        signal = read_signal(os.path.join(
            os.path.expanduser('~'), 'AIChallengeBot', 'new_file.npy'))[0]
        signal = nk.ecg_clean(signal)
        
        meta = message.text.split()
        assert(len(meta) == 4, "Длина != 4")
        
        bot.reply_to(
            message, "Метаданные получены!\nТеперь вы можете воспользоваться /plot, /genFeatures и /predict")
    except Exception as e:
        print(e)
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")


@bot.message_handler(commands=["plot"])
def send_welcome(message):
    try:
        nk.signal_plot(signal)
        fig = plt.gcf()
        fig.savefig("myfig.png")

        img = open('myfig.png', 'rb')
        bot.send_photo(message.chat.id, img, "График ЭКГ сигнала!😎",
                       reply_to_message_id=message.message_id)
        img.close()
    except Exception as e:
        print(e)
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")


@bot.message_handler(commands=["genFeatures"])
def send_welcome(message):
    try:
        bot.reply_to(message, my_features(signal).__str__())
    except Exception as e:
        print(e)
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")


@bot.message_handler(commands=["predict"])
def send_welcome(message):
    try:
        global meta
        signal = read_signal(os.path.join(os.path.expanduser('~'), 'AIChallengeBot', 'new_file.npy'))
        meta_columns = ['age', 'sex', 'height', 'weight', 'record_name']

        dataset = pd.DataFrame({
            'record_name': ['new_file'],
            'signal_0': [clean_signal(signal, 0)],
            'signal_1': [clean_signal(signal, 1)],
            'age': [np.nan if meta[0] == '-' else int(meta[0])],
            'sex': [np.nan if meta[1] == '-' else int(meta[1])],
            'height': [np.nan if meta[2] == '-' else float(meta[2])],
            'weight': [np.nan if meta[3] == '-' else float(meta[3])],
        })
        res = predict(dataset).to_dict()
        
        # for label, checkbox in res.items():
        #     print(label, checkbox[0])
        
        response = ""
        if res['норма'][0] == 1:
            response = "Здоров! ✅"
        else:
            response = "Обнаружены проблемы: " \
            + ", ".join([label.capitalize() for label, checkbox in res.items() if checkbox[0] == 1]) \
            + "❌"
        bot.reply_to(message, response)
    except Exception as e:
        print(e)
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Неизвестная команда! 🤬\nВоспользуйтесь /help")

print("Bot started")
bot.infinity_polling()
