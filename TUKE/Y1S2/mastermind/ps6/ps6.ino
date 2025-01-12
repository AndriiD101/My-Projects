#include "lcd_wrapper.h"
#include "mastermind.h"

void setup() {
  Serial.begin(9600);
  randomSeed(analogRead(A1));

  for(int i=2;i<6;i++){
    pinMode(i,INPUT);
  }
  pinMode(A0,INPUT);

  for(int i=6;i<14;i++){
    pinMode(i,OUTPUT);
  }
  pinMode(BTN_ENTER_PIN,INPUT);
  pinMode(BTN_1_PIN,INPUT);
  pinMode(BTN_2_PIN,INPUT);
  pinMode(BTN_3_PIN,INPUT);
  pinMode(BTN_4_PIN,INPUT);
  lcd_init();
}


void loop() {
  char* code = generate_code(false, 4);
  code = "1342";
  play_game(code);
  free(code);
}