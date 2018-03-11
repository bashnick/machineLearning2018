# -*- coding: utf-8 -*-
import pickle
import re
from flask import Flask, render_template, request
import logging
from logging.handlers import RotatingFileHandler


# Создаём  экземпляр flask-a
app = Flask(__name__)

# Адрес файла с моделью
filename_1 = 'resources/model.sav'
filename_2 = 'resources/vectorizer.sav'

# Создаём эксземпляр нашей модели
model = pickle.load(open(filename_1, 'rb'))
vectorizer = pickle.load(open(filename_2, 'rb'))

# Парсинг формы
def parse_form(form):
    review_text = form.get('review')
    return {
        'review': [re.sub("[,.!:(){}/\-_']+", ' ', review_text.lower())]
    }

# Основная функция, от flask-a нужен декоратор
@app.route("/score", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    # GET запрос - просто получение кода страницы - возвращем то, что есть
    if request.method == "GET":
        return render_template('hello.html')

    # POST запрос - получение кода страницы, но с учётом дополнительных посылаемых данных
    if request.method == "POST":
        # Извлекаем данные и парсим
        app.logger.info('POST request, start to parse data')
        obj = parse_form(request.form)
        app.logger.info('Data is parsed, the result is')
        app.logger.info(obj)
        # Делаем предсказание
        X = vectorizer.transform(obj['review'])
        prediction = model.predict(X)
        pred_text = 'положительная' if prediction > 0.0 else 'отрицательная'
        app.logger.info('The prediction is {}'.format(prediction))
        # Возвращаем страницу с правильно заполненными значениями
        return render_template(
            'hello.html', 
            review=str(obj['review'][0]),
            prediction=pred_text
        )

if __name__ == "__main__":
    # Правильный способ логгировать данные - библиотека logging
 #   formatter = logging.Formatter("[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
 #   handler = RotatingFileHandler('demo.log', maxBytes=10000, backupCount=5)
 #   handler.setLevel(logging.INFO)
 #   handler.setFormatter(formatter)
 #   app.logger.addHandler(handler)
    app.run(host='::', port=80, debug=True)
