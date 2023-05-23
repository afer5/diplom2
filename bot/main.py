import re
import string
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from telegram.ext import Updater, CommandHandler, MessageHandler
from telegram.ext import Filters
from telegram import ReplyKeyboardMarkup
from nudenet import NudeClassifier
from joblib import load   #
from translate import Translator
import speech_recognition as sr #
import os
from pydub import AudioSegment

snowball = SnowballStemmer(language="russian")
russian_stop_words = stopwords.words("russian")


def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens


def tokenize_sentence_without_stopwords(x):
    return tokenize_sentence(x, remove_stop_words=True)


model_pipeline_c_10 = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=tokenize_sentence_without_stopwords, token_pattern=None)),
    ("model", LogisticRegression(random_state=0, C=10.))
])
# load the model from disk
model_pipeline_c_10 = load("../toxic_model/model_end.joblib")

# Define the keyboard
keyboard = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

user_data = {}

# Define global variables to store user's selected language and filtering mode
filter_russia = ""  # Увімкненна російська токсична фільтрація(0,1)
filter_photo = ""  # Увімкненна фільтрація фото(0,1)
filter_audio = ""  # Увімкненна фільтрація аудіо(0,1)
filter_lag = ""  # Увімкненна українська лагідна фільтрація(0,1)
filter_toxic_ua = ""  # Увімкненна українська токсична фільтрація(0,1)
filter_english = ""  # Увімкненна англійська токсична фільтрація(0,1)
language = ""  # Використована мова при аудіо фільтрації(uk,ru,en)


# Define a function to handle the /start command
def start(update, context):
    # Get the user ID or chat ID
    user_id = update.message.chat_id

    # Initialize the user-specific variables if they don't exist yet
    if user_id not in user_data:
        user_data[user_id] = {"filter_russia": 0, "filter_photo": 0, "filter_audio": 0, "filter_lag": 0,
                              "filter_toxic_ua": 0, "filter_english": 0, "language": 0}
    context.bot.send_message(chat_id=update.message.chat_id,
                             text="Вітаю, це бот-модератор. Він допоможе вам видаляти токсичний контент. Оберіть "
                                  "бажану фільтрацію, або натисніть 'Довідка' для додаткової інформації",
                             reply_markup=reply_markup)


# Define a function to show buttons and write text
def synopsis(update, context):
    buttons = [['Текст', 'Аудіо', 'Фото']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text='Цей бот може фільтрувати текст(для більш коректної '
                                                                  'роботи оберіть пріорітетну мову) на наявність'
                                                                  'токсичності,'
                                                                  'аудіо-повідомлення, фото та відео. Текст може '
                                                                  'фільтрувати за трьома мовами: Українська(загальним '
                                                                  'чином, лагідна),Англійська,Російська.Аудіо також '
                                                                  'фільтрує за трьома мовами, але потрібно обрати '
                                                                  'одну у текстовій фільтрації.Медіа-контент перевіряє на наявність 18+ '
                                                                  'контенту та NSFW', reply_markup=reply_markup)


def text(update, context):
    buttons = [['Українська', 'Англійська', 'Російська'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text='Оберіть мову фільтрування.',
                             reply_markup=reply_markup)


def back(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Ви повернулися", reply_markup=reply_markup)


def photo(update, context):
    buttons = [['Ввімкнути фільтр фото', 'Вимкнути фільтр фото'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Увімкнути фільтр фото?", reply_markup=reply_markup)


def photo_on(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_photo"] = 1
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Фільтр фото увімкнений", reply_markup=reply_markup)


def photo_off(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_photo"] = 0
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Фільтр фото вимкнений", reply_markup=reply_markup)


def audio(update, context):
    buttons = [['Ввімкнути', 'Вимкнути'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Увімкнути фільтр аудіо-повідомлень?",
                             reply_markup=reply_markup)


def filter_text_russia(update, context):
    buttons = [['Ввімкнути російський фільтр', 'Вимкнути російський фільтр'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Оберіть необхідну дію", reply_markup=reply_markup)


def filter_text_russia_on(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_lag"] = 0
    user_data[user_id]["filter_toxic_ua"] = 0
    user_data[user_id]["filter_english"] = 0
    user_data[user_id]["filter_russia"] = 1
    user_data[user_id]["language"] = "ru"
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Фільтр російських повідомлень увімкнений",
                             reply_markup=reply_markup)


def filter_text_russia_off(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_russia"] = 0
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Фільтр російських повідомлень вимкнений",
                             reply_markup=reply_markup)


def Ukraine(update, context):
    buttons = [['Лагідна', 'Токсична']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Оберіть режим фільтрування. Лагідна - видалення "
                                                                  "повідомлень з використанням виключно російських "
                                                                  "літер. Токсична - видалення токсичних "
                                                                  "повідомлень. Працювати може тільки один режим.",
                             reply_markup=reply_markup)


def english_text(update, context):
    buttons = [['Ввімкнути англійський фільтр', 'Вимкнути англійський фільтр'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Оберіть необхідну дію", reply_markup=reply_markup)


def english_text_on(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_english"] = 1
    user_data[user_id]["filter_russia"] = 0
    user_data[user_id]["filter_toxic_ua"] = 0
    user_data[user_id]["filter_lag"] = 0
    user_data[user_id]["language"] = "en"
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id,
                             text="Токсична фільтрація англійських повідомлень увімкнена", reply_markup=reply_markup)


def english_text_off(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_english"] = 0
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id,
                             text="Токсична фільтрація англійських повідомлень вимкнена", reply_markup=reply_markup)


def Ukraine_toxic(update, context):
    buttons = [['Ввімкнути український фільтр', 'Вимкнути український фільтр'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Оберіть необхідну дію", reply_markup=reply_markup)


def Ukraine_toxic_on(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_english"] = 0
    user_data[user_id]["filter_russia"] = 0
    user_data[user_id]["filter_toxic_ua"] = 1
    user_data[user_id]["language"] = "uk"
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Токсична фільтрація увімкнена",
                             reply_markup=reply_markup)


def Ukraine_toxic_off(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_toxic_ua"] = 0
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Токсична фільтрація увімкнена",
                             reply_markup=reply_markup)


def Ukraine_lagidna(update, context):
    buttons = [['Ввімкнути лагідний фільтр', 'Вимкнути лагідний фільтр'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Оберіть необхідну дію", reply_markup=reply_markup)


def Ukraine_lagidna_on(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_english"] = 0
    user_data[user_id]["filter_russia"] = 0
    user_data[user_id]["filter_lag"] = 1
    user_data[user_id]["language"] = "uk"
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Лагідна фільтрація увімкнена",
                             reply_markup=reply_markup)


def Ukraine_lagidna_off(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_lag"] = 0
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Лагідна фільтрація вимкнена",
                             reply_markup=reply_markup)


# Define a function to handle messages


def handle_message(update, context):
    user_id = update.message.chat_id
    if user_data[user_id]["filter_lag"] == 1:
        delete_message_lag(update, context)
    if user_data[user_id]["filter_russia"] == 1:
        text = update.message.text
        if model_pipeline_c_10.predict([text]) == [1]:
            delete_message_text(update, context)
    if user_data[user_id]["filter_toxic_ua"] == 1 or user_data[user_id]["filter_english"] == 1:
        text = update.message.text
        if text is not None:
            translator = Translator(to_lang="ru", from_lang=user_data[user_id]['language'])
            translated = translator.translate(text)
            lowercase_text = translated.lower()
            if model_pipeline_c_10.predict([lowercase_text]) == [1]:
                delete_message_text(update, context)



def delete_message_lag(update, context):
    text = update.message.text
    if re.search('[ыъэё]', text, re.IGNORECASE):
        # Delete the message
        context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)
        # Send a notification to the user that their message was deleted
        context.bot.send_message(chat_id=update.message.chat_id, text="Ваше повідомлення було видалено через наявність "
                                                                      "російських літер.")


def delete_message_text(update, context):
    # Delete the message
    context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)
    # Send a notification to the user that their message was deleted
    context.bot.send_message(chat_id=update.message.chat_id, text="Ваше повідомлення було токсичне.")


# define a function to delete a photo message if it's a nude photo
def detect_nude_photo(photo):
    c = NudeClassifier()
    data = c.classify(photo)
    if data[photo]['safe'] < 0.9:
        return 1  # nude
    else:
        return 0  # not nude


def delete_photo(update, context):
    user_id = update.message.chat_id
    if user_data[user_id]["filter_photo"] == 1:
        # Get the chat ID and message ID of the incoming message
        chat_id = update.message.chat_id
        message_id = update.message.message_id

        # Download the photo to disk
        photo_file = context.bot.get_file(update.message.photo[-1].file_id)
        file_path = os.path.join(os.getcwd(), 'photos', f'{message_id}.jpg')
        photo_file.download(file_path)

        # Check if the photo is unsafe
        is_nude_photo = detect_nude_photo(file_path)

        # Delete the message if the photo is unsafe
        if is_nude_photo:
            context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            context.bot.send_message(chat_id=update.message.chat_id, text="фотокартку було видалено, через токсичний зміст",
                                     reply_markup=reply_markup)
        # Remove the photo file from disk
        os.remove(file_path)


def audio(update, context):
    buttons = [['Ввімкнути аудіо фільтр', 'Вимкнути аудіо фільтр'], ['Назад']]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Для коректної роботи спочатку оберіть мову "
                                                                  "текстового фільтру. Оберіть необхідну дію",
                             reply_markup=reply_markup)


def audio_on(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_audio"] = 1
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Аудіо фільтрація увімкнена",
                             reply_markup=reply_markup)


def audio_off(update, context):
    buttons = [['Текст', 'Аудіо'], ['Фото', 'Довідка']]
    user_id = update.message.chat_id
    user_data[user_id]["filter_audio"] = 0
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    context.bot.send_message(chat_id=update.message.chat_id, text="Аудіо фільтрація вимкнена",
                             reply_markup=reply_markup)





def delete_voice_message(update, context):
    user_id = update.message.chat_id
    if user_data[user_id]["filter_audio"] == 1:
        # Get the chat ID and message ID of the incoming message
        chat_id = update.message.chat_id
        message_id = update.message.message_id
        # Download the voice message to disk
        voice_message_file = context.bot.get_file(update.message.voice.file_id)
        file_path = os.path.join(os.getcwd(), 'voice_messages', f'{message_id}.ogg')
        voice_message_file.download(file_path)
        # Convert the file to PCM WAV format
        wav_file_path = ogg_to_wav(file_path)
        # Transcribe the voice message to text
        text = transcribe_voice_message(wav_file_path, update)
        if language != "ru":
            if text is not None:
                translator = Translator(to_lang="ru", from_lang=user_data[user_id]['language'])
                translated = translator.translate(text)
                text = translated.lower()
        # Check if the message is inappropriate
        is_inappropriate = model_pipeline_c_10.predict([text])
        # Delete the message if it is inappropriate
        if is_inappropriate:
            try:
                context.bot.delete_message(chat_id=chat_id, message_id=message_id)
                context.bot.send_message(chat_id=update.message.chat_id, text="Голосове повідомлення було токсичне",
                                         reply_markup=reply_markup)
            except Exception as e:
                print(f"Failed to delete message {message_id}: {e}")

        # Remove the files from disk
        os.remove(file_path)
        os.remove(wav_file_path)


def transcribe_voice_message(wav_file_path, update):
    user_id = update.message.chat_id
    r = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data,
                                  language=f"{user_data[user_id]['language']}-{user_data[user_id]['language'].upper()}")
    return text


def ogg_to_wav(file_path):
    sound = AudioSegment.from_ogg(file_path)
    wav_file_path = os.path.splitext(file_path)[0] + '.wav'
    sound.export(wav_file_path, format="wav")
    return wav_file_path


# Set up the bot
updater = Updater(token="5971758778:AAECLGnBvQEq6GOtDCv-LYA9PpDbDqa8ld8", use_context=True)
dispatcher = updater.dispatcher

# Register the handlers
start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

audio_button_handler = MessageHandler(Filters.regex(re.compile('^аудіо$', re.IGNORECASE)), audio)
dispatcher.add_handler(audio_button_handler)

audio_on_handler = MessageHandler(Filters.regex(re.compile('^ввімкнути аудіо фільтр$', re.IGNORECASE)), audio_on)
dispatcher.add_handler(audio_on_handler)

audio_off_handler = MessageHandler(Filters.regex(re.compile('^вимкнути аудіо фільтр$', re.IGNORECASE)), audio_off)
dispatcher.add_handler(audio_off_handler)

voice_handler = MessageHandler(Filters.voice, delete_voice_message)
dispatcher.add_handler(voice_handler)

synopsis_handler = MessageHandler(Filters.regex(re.compile('^довідка$', re.IGNORECASE)), synopsis)
dispatcher.add_handler(synopsis_handler)

english_text_handler = MessageHandler(Filters.regex(re.compile('^англійська$', re.IGNORECASE)), english_text)
dispatcher.add_handler(english_text_handler)

english_text_on_handler = MessageHandler(Filters.regex(re.compile('^Ввімкнути англійський фільтр$', re.IGNORECASE)),
                                         english_text_on)
dispatcher.add_handler(english_text_on_handler)

english_text_off_handler = MessageHandler(Filters.regex(re.compile('^Вимкнути англійський фільтр$', re.IGNORECASE)),
                                          english_text_off)
dispatcher.add_handler(english_text_off_handler)

russia_text_handler = MessageHandler(Filters.regex(re.compile('^російська$', re.IGNORECASE)), filter_text_russia)
dispatcher.add_handler(russia_text_handler)

russia_text_on_handler = MessageHandler(Filters.regex(re.compile('^ввімкнути російський фільтр$', re.IGNORECASE)),
                                        filter_text_russia_on)
dispatcher.add_handler(russia_text_on_handler)

russia_text_off_handler = MessageHandler(Filters.regex(re.compile('^вимкнути російський фільтр$', re.IGNORECASE)),
                                         filter_text_russia_off)
dispatcher.add_handler(russia_text_off_handler)

Ukraine_handler = MessageHandler(Filters.regex(re.compile('^українська$', re.IGNORECASE)), Ukraine)
dispatcher.add_handler(Ukraine_handler)

Ukraine_toxic_handler = MessageHandler(Filters.regex(re.compile('^токсична$', re.IGNORECASE)), Ukraine_toxic)
dispatcher.add_handler(Ukraine_toxic_handler)

Ukraine_toxic_on_handler = MessageHandler(Filters.regex(re.compile('^ввімкнути український фільтр$', re.IGNORECASE)),
                                          Ukraine_toxic_on)
dispatcher.add_handler(Ukraine_toxic_on_handler)

Ukraine_toxic_off_handler = MessageHandler(Filters.regex(re.compile('^вимкнути український фільтр$', re.IGNORECASE)),
                                           Ukraine_toxic_off)
dispatcher.add_handler(Ukraine_toxic_off_handler)

Ukraine_lagidna_handler = MessageHandler(Filters.regex(re.compile('^лагідна$', re.IGNORECASE)), Ukraine_lagidna)
dispatcher.add_handler(Ukraine_lagidna_handler)

Ukraine_lagidna_handler_on = MessageHandler(Filters.regex(re.compile('^Ввімкнути лагідний фільтр$', re.IGNORECASE)),
                                            Ukraine_lagidna_on)
dispatcher.add_handler(Ukraine_lagidna_handler_on)

Ukraine_lagidna_handler_off = MessageHandler(Filters.regex(re.compile('^Вимкнути лагідний фільтр$', re.IGNORECASE)),
                                             Ukraine_lagidna_off)
dispatcher.add_handler(Ukraine_lagidna_handler_off)

audio_handler = MessageHandler(Filters.regex(re.compile('^аудіо$', re.IGNORECASE)), audio)
dispatcher.add_handler(audio_handler)

photo_handler = MessageHandler(Filters.regex(re.compile('^фото$', re.IGNORECASE)), photo)
dispatcher.add_handler(photo_handler)

photo_on_handler = MessageHandler(Filters.regex(re.compile('^ввімкнути фільтр фото$', re.IGNORECASE)), photo_on)
dispatcher.add_handler(photo_on_handler)

photo_off_handler = MessageHandler(Filters.regex(re.compile('^вимкнути фільтр фото$', re.IGNORECASE)), photo_off)
dispatcher.add_handler(photo_off_handler)

photo_handler = MessageHandler(Filters.photo, delete_photo)
dispatcher.add_handler(photo_handler)

text_handler = MessageHandler(Filters.regex(re.compile('^текст$', re.IGNORECASE)), text)
dispatcher.add_handler(text_handler)

back_handler = MessageHandler(Filters.regex(re.compile('^назад$', re.IGNORECASE)), back)
dispatcher.add_handler(back_handler)

message_handler = MessageHandler(Filters.text, handle_message)
dispatcher.add_handler(message_handler)

# Start the bot
updater.start_polling()
updater.idle()

