#include <superkarel.h>

void turn_right()
{
    turn_left();
    turn_left();
    turn_left();
}

void turn_around()
{
    turn_left();
    turn_left();
}

void jump_over()
{
    if(!right_is_blocked())
        turn_right();
    else if(!left_is_blocked() && front_is_blocked())
        turn_left();
    else if(left_is_blocked() && front_is_blocked() && right_is_blocked())
        turn_around();
}

int main()
{
    set_step_delay(100);
    turn_on("task_1.kw");
    if(beepers_present())
    {
        pick_beeper();
        turn_off();
    }
    else
    {
    put_beeper();
    while(front_is_clear())
        step();
    turn_left();
    step();
    while(!beepers_present())
    {
        jump_over();
        step();
    }
    if(beepers_present())
        pick_beeper();
    while(!beepers_present())
    {
        jump_over();
        step();
    }
    if(beepers_present())
        pick_beeper();
    while(!facing_west())
        turn_left();
    }
    turn_off();
    return 0;
}

