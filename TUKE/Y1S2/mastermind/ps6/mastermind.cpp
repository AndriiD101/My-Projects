#include "mastermind.h"
#include "lcd_wrapper.h"
#include <stdbool.h>
#include <Arduino.h>
#include <assert.h>

char* generate_code(bool repeat, int length) {
  if (length < 1 || length > 10) {
    return NULL;
  }

  char* secret = (char*) malloc((length + 1) * sizeof(char));
  
  if (repeat) {
    for (int i = 0; i < length; i++) {
      secret[i] = '0' + random(10);
    }
  } else {
    for (int i = 0; i < length; i++) {
      bool unique;
      int randNum;
      do {
        unique = true;
        randNum = random(10);
        for (int j = 0; j < i; j++) {
          if (secret[j] == '0' + randNum) {
            unique = false;
            break;
          }
        }
      } while (!unique);
      secret[i] = '0' + randNum;
    }
  }
  
  secret[length] = '\0';
  return secret;
}



void get_score(const char* secret, const char* guess, int* peg_a, int* peg_b) {
  int len_secret = strlen(secret);
  int len_guess = strlen(guess);

  if (len_secret != len_guess) return;

  *peg_a = 0;
  *peg_b = 0;

  int secret_freq[10] = {0};
  int guess_freq[10] = {0};

  for (int i = 0; i < len_secret; i++) {
    if (secret[i] == guess[i]) (*peg_a)++;
    else {
      secret_freq[secret[i] - '0']++;
      guess_freq[guess[i] - '0']++;
    }
  }

  for (int i = 0; i < 10; i++) {
    if (secret_freq[i] < guess_freq[i]) {
      (*peg_b) += secret_freq[i];
    } else {
      (*peg_b) += guess_freq[i];
    }
  }
}



void render_leds(const int peg_a, const int peg_b) {

  int blue_pin = LED_BLUE_1; 
  int red_pin = LED_RED_1;  

  if (peg_a > 0  && peg_b == 0) {
    for (int i = 0; i < peg_a; i++) {
      digitalWrite(blue_pin, HIGH);
      blue_pin = blue_pin + 2;
    }
  }
  if (peg_b > 0 && peg_a == 0) {
    for (int i = 0; i < peg_b; i++) {
      digitalWrite(red_pin, HIGH);
      red_pin = red_pin + 2;
    }
  }
  if(peg_a>0 && peg_b>0){
    int current_led = 6;
    for(int i =0; i<peg_a; i++){
      digitalWrite(current_led, HIGH);
      current_led+=2;
    }
    current_led++;
    for(int i =0; i<peg_b; i++){
      digitalWrite(current_led, HIGH);
      current_led+=2;
    }
  }
}

void turn_off_leds() {
  for (int i = 6; i <= 13; i++) {
    digitalWrite(i, LOW);
  }
}

void render_history(char* secret, char** history, const int entry_nr) {
  lcd_print_at(1,0,history[entry_nr]);
  int peg_a = 0;
  int peg_b = 0;
  get_score(secret, history[entry_nr], &peg_a, &peg_b);
  turn_off_leds();
  render_leds(peg_a, peg_b);
}

void play_game(char* secret) {
  //VARIABLES
  int tries = 0;
  char* history[10];
  int history_added = 0;
  int history_index = 0;  
  int peg_a, peg_b;
  char tmp_try[2] = {'0', '\0'};
  bool pressed_enter;
  bool is_lost = false;
  // char array_guess[5]={'0', '0', '0', '0', '\0'};
  //START GAME
  turn_off_leds();
  lcd_clear(); 
  lcd_print_at(0, 0, "Mastermind!");
  lcd_print_at(1, 0, "Guess sequence");
  delay(3000);
  lcd_clear();
  //SHOULD BE COMMENTED
  // lcd_print_at(1, 0, secret);
  // delay(2000);
  // lcd_clear();
  //-------------------
  
  for (int i = 0; i < 10; i++) {
    history[i] = (char*)malloc(5 * sizeof(char));
  }

  while(tries <= 9) {
    char array_guess[5]={'0', '0', '0', '0', '\0'};
    tmp_try[0] = tries+'0';
    lcd_clear();
    lcd_print(array_guess); 
    lcd_print_at(1, 0, "your attampt:");
    lcd_print_at(1, 13, tmp_try);
    pressed_enter = false;
    
    while(!pressed_enter) {
      if(digitalRead(BTN_1_PIN) == HIGH){
        if(array_guess[0] < '9') {
          array_guess[0]++;
        } else {
          array_guess[0] = '0';
        }
        delay(250);
        lcd_clear();
        lcd_print(array_guess); 
        lcd_print_at(1, 0, "your attampt:");
        lcd_print_at(1, 13, tmp_try); 
        delay(300);
        if(digitalRead(BTN_1_PIN) == HIGH && digitalRead(BTN_2_PIN) == HIGH){
          lcd_clear();
          lcd_print_at(0, 0, "HISTORY UP");
          if(history_added == 0) {
            lcd_print_at(1, 0, "no history");
          }else{
            if(history_index < history_added - 1) {  
              history_index++;
            }
            render_history(secret, history, history_index);
          }
          delay(1800);
        }
        else if(digitalRead(BTN_1_PIN) == HIGH && digitalRead(BTN_3_PIN) == HIGH){
          lcd_clear();
          lcd_print_at(0, 0, "HISTORY DOWN");
          if(history_added == 0) {
            lcd_print_at(1, 0, "no history");
          }else{
            if(history_index > 0) {  
              history_index--;
            }
            render_history(secret, history, history_index);
          }
          delay(1800);
        }
      }

      //button 2
      if(digitalRead(BTN_2_PIN) == HIGH){
        if(array_guess[1] < '9') {
          array_guess[1]++;
        } else {
          array_guess[1] = '0';
        }
        delay(250);
        lcd_clear();
        lcd_print(array_guess); 
        lcd_print_at(1, 0, "your attampt:");
        lcd_print_at(1, 13, tmp_try); 
      }

      //button 3
      if(digitalRead(BTN_3_PIN) == HIGH){
        if(array_guess[2] < '9') {
          array_guess[2]++;
        } else {
          array_guess[2] = '0';
        }
        delay(250);
        lcd_clear();
        lcd_print(array_guess); 
        lcd_print_at(1, 0, "your attampt:");
        lcd_print_at(1, 13, tmp_try); 
      }

      //button 4
      if(digitalRead(BTN_4_PIN) == HIGH){
        if(array_guess[3] < '9') {
          array_guess[3]++;
        } else {
          array_guess[3] = '0';
        }
        delay(250);
        lcd_clear();
        lcd_print(array_guess); 
        lcd_print_at(1, 0, "your attampt:");
        lcd_print_at(1, 13, tmp_try); 
      }

      //enter button
      if(digitalRead(BTN_ENTER_PIN) == HIGH){
        pressed_enter = true;
        for(int i = 0; i < 4; i++){
          history[tries][i] = array_guess[i];
        }
        history[tries][4] = '\0';  
        history_added++;
        break; 
      }
    }
    delay(250);
    turn_off_leds();
    get_score(secret, array_guess, &peg_a, &peg_b);

    char str_peg_a[10];
    char str_peg_b[10];

    sprintf(str_peg_a, "%d", peg_a);
    sprintf(str_peg_b, "%d", peg_b);

    lcd_clear();
    lcd_print_at(0, 0, "NUM & PLACE: ");
    lcd_print_at(0, 13, str_peg_a);

    lcd_print_at(1, 0, "ONLY NUM: ");
    lcd_print_at(1, 10, str_peg_b);

    render_leds(peg_a, peg_b);
    delay(1000);

    if(peg_a == 4 && peg_b == 0){
      turn_off_leds();
      lcd_clear();
      lcd_print_at(0, 0, "Congrats");
      lcd_print_at(1,0, "You WON!!!");
      for(int i = 6; i<14; i++){
        delay(250);
        digitalWrite(i, HIGH);
      }
      delay(1000);
      turn_off_leds();
      for(int i = 6; i<14; i++){
        delay(250);
        digitalWrite(i, HIGH);
      }
      delay(1000);
      turn_off_leds();
      for(int i = 6; i<14; i++){
        delay(250);
        digitalWrite(i, HIGH);
      }
      delay(100);
      turn_off_leds();
      break;
    }
    delay(2000);
    tries++;
    if(tries==9) is_lost = true;
  }
  if(is_lost == true){
  turn_off_leds();
  lcd_clear();
  lcd_print_at(0, 0, "Unfortunetely");
  lcd_print_at(1, 0, "You LOST!!!(((");
  for(int i = LED_RED_1; i<=LED_RED_4; i+=2){
    delay(250);
    digitalWrite(i, HIGH);
  }
  delay(1000);
  turn_off_leds();
  for(int i = LED_RED_1; i<=LED_RED_4; i+=2){
    delay(250);
    digitalWrite(i, HIGH);
  }
  delay(1000);
  turn_off_leds();
  for(int i = LED_RED_1; i<=LED_RED_4; i+=2){
    delay(250);
    digitalWrite(i, HIGH);
  }
  delay(1000);
  turn_off_leds();
  }
}
