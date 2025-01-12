#include <Arduino.h>
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);
#include "lcd_wrapper.h"

void lcd_init(){
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0,0);
}

void lcd_clear(){
	lcd.clear();
}


void lcd_set_cursor(int y, int x){
	lcd.setCursor(x,y);
}


void lcd_print(char* text){
    lcd.print(text);
}


void lcd_print_at(int y, int x, char* text){
    lcd_set_cursor(y, x);
    lcd_print(text);
}
