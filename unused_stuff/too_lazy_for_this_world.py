import subprocess
import os
import webbrowser
import datetime
import time
from datetime import timedelta


week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]



w=time.strftime("%w", time.localtime())
h=time.strftime("%H", time.localtime())
m=time.strftime ("%M", time.localtime())
s=time.strftime ("%S", time.localtime())



def lessos_Monday(h, m, s):
    total_seconds = h * 3600 + m * 60 + s
    while total_seconds > 0:
        timer = datetime.timedelta(seconds=total_seconds)
        print(timer, end="\r")
        if total_seconds == 32400:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0NjE1NjkwODU4")
            webbrowser.open("https://meet.google.com/okm-xkmp-utv?authuser=1")
        elif total_seconds == 39600:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0OTk2NTc4NTk3")
            webbrowser.open("https://meet.google.com/zjb-aueg-qgs?authuser=1")
        elif total_seconds == 43380:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0NjM5OTY0NTA4")
            webbrowser.open("https://mokyklelepasaka.zoom.us/j/89248423650?pwd=M3h3UDcrVGdUeDlKODMrUm1XWW5mUT09#success")
        elif total_seconds == 46980:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0OTUwMzg3Njkz")
            webbrowser.open("https://meet.google.com/bzj-cqvg-hyu?authuser=1&hs=179")
            webbrowser.open("https://us04web.zoom.us/j/71072525741?pwd=f9ZjoJiEjxGdaG0XpWLoN4jPwqB10k.1")
            return True
        time.sleep(1)
        total_seconds += 1

def lessons_Tuesday(h, m, s):
    total_seconds = h * 3600 + m * 60 + s
    while total_seconds > 0:
        timer = datetime.timedelta(seconds=total_seconds)
        print(timer, end="\r")
        if total_seconds == 32400:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0NjE1NjkwODU4")
            webbrowser.open("https://meet.google.com/okm-xkmp-utv?authuser=1")
        elif total_seconds == 39600:
            webbrowser.open('https://classroom.google.com/u/1/c/NTQ0NDY2ODAxNDM2')
            webbrowser.open("https://meet.google.com/zag-jkbn-xfa?authuser=1")
        elif total_seconds == 43380:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0NjM5OTY0NTA4")
            webbrowser.open("https://mokyklelepasaka.zoom.us/j/89248423650?pwd=M3h3UDcrVGdUeDlKODMrUm1XWW5mUT09#success")
            return True
        time.sleep(1)
        total_seconds += 1

def lessons_Wednesday(h, m, s):
    total_seconds = h * 3600 + m * 60 + s
    while total_seconds > 0:
        timer = datetime.timedelta(seconds=total_seconds)
        print(timer, end="\r")
        if total_seconds == 32400:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0OTc3MTE3OTUz")
            webbrowser.open("https://meet.google.com/roz-mnma-hrw?authuser=1")
        elif total_seconds == 39600:
            webbrowser.open('https://classroom.google.com/u/1/c/MjQ5MTE0MTI0MDM5')
            webbrowser.open("https://meet.google.com/fbc-ydgd-nmm?authuser=1")
        elif total_seconds == 43380:
            webbrowser.open("https://classroom.google.com/u/1/c/NDIxODM1MzU4MTEw")
            webbrowser.open("https://meet.google.com/wev-nxaz-xxo?authuser=1")
        elif total_seconds == 50400:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0OTk1NjA5NzE4")
            webbrowser.open("https://meet.google.com/uyh-wdfa-imw?authuser=1")
            return True
        time.sleep(1)
        total_seconds += 1

def lessons_Thursday(h, m, s):
    total_seconds = h * 3600 + m * 60 + s
    while total_seconds > 0:
        timer = datetime.timedelta(seconds=total_seconds)
        print(timer, end="\r")
        if total_seconds == 32400:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0NjE1NjkwODU4")
            webbrowser.open("https://meet.google.com/okm-xkmp-utv?authuser=1")
        elif total_seconds == 39600:
            webbrowser.open("https://classroom.google.com/u/1/c/MTg2MjU2Mjc5NDI1")
            webbrowser.open("https://meet.google.com/zzs-pvgj-gqe?authuser=1")
        elif total_seconds == 43380:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0OTc3MTE3OTUz")
            webbrowser.open("https://meet.google.com/roz-mnma-hrw?authuser=1")
        elif total_seconds == 46980:
            webbrowser.open("https://classroom.google.com/u/1/c/Mzk4OTYwNjgxMDc5")
            webbrowser.open("https://meet.google.com/rku-pugx-xem?authuser=1")
        elif total_seconds == 50400:
            webbrowser.open("https://meet.google.com/rhj-gdjx-edc?authuser=1")
            return True
        time.sleep(1)
        total_seconds += 1

def lessons_Friday(h, m, s):
    total_seconds = h * 3600 + m * 60 + s
    while total_seconds > 0:
        timer = datetime.timedelta(seconds=total_seconds)
        print(timer, end="\r")
        if total_seconds == 32400:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0NjM5OTY0NTA4")
            webbrowser.open("https://mokyklelepasaka.zoom.us/j/89248423650?pwd=M3h3UDcrVGdUeDlKODMrUm1XWW5mUT09#success")
        elif total_seconds == 39600:
            webbrowser.open("https://classroom.google.com/u/1/c/NDIxODM1MzU5MzE3")
            webbrowser.open("https://meet.google.com/mor-encb-foh?authuser=1")
        elif total_seconds == 43380:
            webbrowser.open("https://classroom.google.com/u/1/c/NTI2ODQwOTc4MDk5")
            webbrowser.open("https://meet.google.com/khp-txst-cbm?authuser=1")
        elif total_seconds == 46980:
            webbrowser.open("https://classroom.google.com/u/1/c/NTQ0NjE5NzIyNzU0")
            webbrowser.open("https://meet.google.com/erd-iqas-dnu?authuser=1")
            return True
        time.sleep(1)
        total_seconds += 1



while True:
    action = input("what do you want to do study or rest \nIf you want to exit press enter \n>>>: ")
    if action == "study" or action == "Study":
        if w == "1":
            lessos_Monday(int(h), int(m), int(s))
        elif w == "2":
            lessons_Tuesday(int(h), int(m), int(s))
        elif w == "3":
            lessons_Wednesday(int(h), int(m), int(s))
        elif w == "4":
            lessons_Thursday(int(h), int(m), int(s))
        elif w == "5":
            lessons_Friday(int(h), int(m), int(s))
        else:
            print("today is weekend")
    elif action == "rest" or action == "Rest":
        rest = input("Choose one out three: \n1) Listen to music \n2) Youtube \n3) Twitch\n>>>: ")
        if rest == "Listen to music" or rest == "listen to music":
            os.startfile("C:\\Users\\denys\\AppData\\Roaming\\Spotify\\Spotify.exe")
        elif rest == "Youtube" or rest == "youtube":
            webbrowser.open("youtube.com")
        elif rest == "Twitch" or rest == "twitch":
            webbrowser.open("twitch.com")
    elif action=="":
            break
    else:
        print("try one more time")