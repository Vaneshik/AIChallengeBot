from telebot import TeleBot, types
from dotenv import load_dotenv

import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def read_signal(path: str):
    with open(path, mode='rb') as f:
        return np.load(f, allow_pickle=True)


def my_features(ecg_signal, sampling_rate):
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

    r_peaks = nk.ecg_peaks(
        ecg_cleaned, sampling_rate=sampling_rate, correct_artifacts=True)
    ecg_rate = nk.ecg_rate(r_peaks, sampling_rate=sampling_rate)
    ecg_vars = [np.mean(ecg_rate), np.min(ecg_rate), np.max(
        ecg_rate), np.max(ecg_rate) - np.min(ecg_rate)]

    hrv_indices = nk.hrv(r_peaks[0], sampling_rate=sampling_rate)
    needed = hrv_indices[['HRV_MeanNN', 'HRV_SDNN', 'HRV_PIP',
                          'HRV_IALS', 'HRV_PSS', 'HRV_PAS', 'HRV_Cd', 'HRV_Ca']]

    entropy = nk.entropy_sample(ecg_cleaned, 1, 4)[0]

    features = ecg_vars + needed.iloc[0].to_list() + [entropy]

    return features


load_dotenv()
bot_token = os.getenv("BOT_TOKEN")
signal = None
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
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")


def process_ecg_meta(message):
    global signal
    try:
        signal = read_signal(os.path.join(
            os.path.expanduser('~'), 'aiijc_bot', 'new_file.npy'))[0]
        signal = nk.ecg_clean(signal)

        msg = bot.reply_to(
            message, "Метаданные получены!\nТеперь вы можете воспользоваться /plot, /genFeatures и /predict")
    except Exception as e:
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
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")


@bot.message_handler(commands=["genFeatures"])
def send_welcome(message):
    try:
        bot.reply_to(message, my_features(signal, 500).__str__())
    except Exception as e:
        bot.reply_to(
            message, "Произошла неизвестная ошибка, свяжитесь с разработчиком - @Vaneshik")

@bot.message_handler(commands=["predict"])
def send_welcome(message):
    bot.reply_to(message, "*тут будет пердикт*")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Неизвестная команда! 🤬\nВоспользуйтесь /help")


bot.infinity_polling()
