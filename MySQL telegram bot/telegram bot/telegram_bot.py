import telebot
import requests

API_KEY_TELEGRAM = "7291165381:AAGNPCa2jI5V9SimqRYOOe7UKUGElNW78CE"
API_KEY_OPENWEATHER = "d039ca07f1aa96d0f872e02adc2f6d83"

BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
TARGET = "C:/Users/denys/OneDrive/Desktop/Programming/telegram bot/city.txt"

bot = telebot.TeleBot(API_KEY_TELEGRAM)

@bot.message_handler(commands=["greet"])
def greet(message):
    bot.reply_to(message, "Hey! How's it going?")

@bot.message_handler(commands=["weather"], func=lambda msg: True)
def give_weather(message):
    message_tokens = message.text.split(" ")
    
    for city in message_tokens:
        if city != message_tokens[0]:
            url = BASE_URL + "appid=" + API_KEY_OPENWEATHER + "&q=" + city
            response = requests.get(url).json()
            print(response)

            weather_description = response.get("weather")[0].get("description")
            temperature = response.get("main").get("temp")

            formatted_message = f"Weather in {city}: {weather_description}, Temperature: {round(temperature-273.15)}°C"
            bot.reply_to(message, formatted_message)
            with open(TARGET, "a") as cityfile:
                cityfile.write(city + "\n")
            bot.reply_to(message, f"City '{city}' saved!")
            with open(TARGET, "r") as cityfile:
                print(cityfile.read())

@bot.message_handler(commands=["clear"])
def clear_file(message):
    file = open(TARGET, "r+")
    if file.read(1):
        file.truncate(0)
        bot.send_message("1059458014", "History successfully cleared")
        file.close()
    else:
        bot.send_message("1059458014", "History is empty")
        file.close()

@bot.message_handler(commands=["end_comunication"])
def stop(commands):
    bot.send_message("1059458014", "I am turned off")
    bot.stop_bot()

if __name__ == "__main__":
    bot.polling()