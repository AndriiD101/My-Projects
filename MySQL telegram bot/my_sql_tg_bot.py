import telebot
import my_sql_tg_bot
import MySQLdb
from datetime import datetime

BOT_API = "7305061505:AAGB8y1iAMa2s5vbP3Buvw3dBYClfe5VZXY"

BOT = telebot.TeleBot(BOT_API)

@BOT.message_handler(commands=["start"])
async def print_greatings(message):
    BOT.send_message("1059458014", "I have been started!")
    print("I have been started")

if __name__ == '__main__':
    BOT.polling()