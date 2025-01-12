// void play_game(char* secret) {
//     int i = -1, y = -1, z = -1, u = -1;
//     int attempts = 0;

//     bool ENTER = false;
//     bool prevState1 = true, prevState2 = true, prevState3 = true, prevState4 = true, prevState5 = true;
//     unsigned long timeChange1, timeChange2, timeChange3, timeChange4, timeChange5;

//     turn_off_leds();

//     lcd_print_at(0, 0, "Mastermind!");
//     delay(3000);
//     lcd_print_at(0, 0, "Guess password");
//     delay(2000);
//     Serial.println(secret);

//     while(attempts < 9) {
//         ENTER = false;
//         while(ENTER == false) {
//             // Button 1
//             if(digitalRead(BTN_1_PIN) == 0) {
//                 if(prevState1 == true && millis() - timeChange1 > 50) {
//                     i++;
//                     if(i > 9) {
//                         i = 0;
//                     }
//                     char ii[2];
//                     ii[0] = '0' + i;
//                     ii[1] = '\0';
//                     lcd_print_at(1, 0, ii);
//                     Serial.println(i);
//                     prevState1 = false;
//                 }
//             } else {
//                 timeChange1 = millis();
//                 prevState1 = true;
//             }
            
//             // Button 2
//             if(digitalRead(BTN_2_PIN) == 0) {
//                 if(prevState3 == true && millis() - timeChange3 > 50) {
//                     y++;
//                     if(y > 9) {
//                         y = 0;
//                     }
//                     char yy[2];
//                     yy[0] = '0' + y;
//                     yy[1] = '\0';
//                     lcd_print_at(1, 0, yy);
//                     Serial.println(y);
//                     prevState3 = false;
//                 }
//             } else {
//                 timeChange3 = millis();
//                 prevState3 = true;
//             }

//             // Button 3
//             if(digitalRead(BTN_3_PIN) == 0) {
//                 if(prevState4 == true && millis() - timeChange4 > 50) {
//                     z++;
//                     if(z > 9) {
//                         z = 0;
//                     }
//                     char zz[2];
//                     zz[0] = '0' + z;
//                     zz[1] = '\0';
//                     lcd_print_at(1, 0, zz);
//                     Serial.println(z);
//                     prevState4 = false;
//                 }
//             } else {
//                 timeChange4 = millis();
//                 prevState4 = true;
//             }

//             // Button 4
//             if(digitalRead(BTN_4_PIN) == 0) {
//                 if(prevState5 == true && millis() - timeChange5 > 50) {
//                     u++;
//                     if(u > 9) {
//                         u = 0;
//                     }
//                     char uu[2];
//                     uu[0] = '0' + u;
//                     uu[1] = '\0';
//                     lcd_print_at(1, 0, uu);
//                     Serial.println(u);
//                     prevState5 = false;
//                 }
//             } else {
//                 timeChange5 = millis();
//                 prevState5 = true;
//             }

//             // Confirm button
//             if(digitalRead(BTN_ENTER_PIN) == 0) {
//                 if(prevState2 == true && millis() - timeChange2 > 50) {
//                     prevState2 = false;
//                 } 
//             } else {
//                 timeChange2 = millis();
//                 prevState2 = true;
//                 turn_off_leds();
//                 ENTER = true;
//             } 
//         } 

//         // Code for guessing and checking the guess
//         char* guess = (char*)calloc(strlen(secret),sizeof(char));
//         char number = '0';
//         char randomnumber;

//         randomnumber = number + i;
//         guess[0] = randomnumber;

//         randomnumber = number + y;
//         guess[1] = randomnumber;

//         randomnumber = number + z;
//         guess[2] = randomnumber;

//         randomnumber = number + u;
//         guess[3] = randomnumber;
//         guess[strlen(secret)] = '\0';    

//         int peg_a;
//         int peg_b;

//         get_score(secret, guess, &peg_a, &peg_b);
//         render_leds(peg_a, peg_b);
//         delay(1000);

//         if (peg_a == 4) { 
//             // Code for winning...
//             lcd_clear();
//             lcd_print_at(0, 0, "YOU WON !");
//             digitalWrite(6, HIGH);
//             digitalWrite(7, HIGH);
//             digitalWrite(8, HIGH);
//             digitalWrite(9, HIGH);
//             digitalWrite(10, HIGH);
//             digitalWrite(11, HIGH);
//             digitalWrite(12, HIGH);
//             digitalWrite(13, HIGH);
//             delay(5000);
//             attempts = 10;
//             turn_off_leds();
//             peg_a = 0;
//             peg_b = 0;
//         } else {
//             // Code for not winning...
//             delay(2000);
//             Serial.println(i);
//             attempts++;
//             char peg_aa[3];
//             peg_aa[0] = '0' + peg_a;
//             peg_aa[1] = '\0';
//             char attemptNumber[3];
//             attemptNumber[0] = '0' + attempts;
//             attemptNumber[1] = '\0';
//             char peg_bb[3];
//             peg_bb[0] = '0' + peg_b;
//             peg_bb[1] = '\0';
//             lcd_clear();
//             lcd_print_at(0, 0, "Correct number");
//             lcd_print_at(1, 0, peg_bb);
//             delay(2000);
//             lcd_clear();
//             lcd_print_at(0, 0, "Correct position");
//             lcd_print_at(1, 0, peg_aa);
//             delay(2000);
//             lcd_clear();
//             lcd_print_at(0, 0, "Not guessed");
//             delay(2000);
//             lcd_clear();
//             turn_off_leds();
//             lcd_print_at(0, 0, "Attempt number :");
//             lcd_print_at(1, 0, attemptNumber);
//             delay(2000);
//             lcd_print_at(0, 0, "Guess password");
//             i = 0; 
//             y = 0;
//             z = 0;
//             u = 0;
//         } 
//     }
//     lcd_clear();
//     lcd_print_at(0, 0, "END !!");
//     turn_off_leds();
//     delay(5000);
// }
