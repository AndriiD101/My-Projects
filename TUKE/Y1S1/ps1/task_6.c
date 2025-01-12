#include <superkarel.h>

void turn_right()
{
    turn_left();
    turn_left();
    turn_left();
}

void move_to_beepers()
{
     while(!beepers_present())
       step();
}

void take_and_turn_exit()
{
    if(beepers_present())
    {
        pick_beeper();
        turn_left();
        take_and_turn_exit();
    }
    if(facing_north())
    {
       while(!facing_east())
           turn_left();
       turn_off();
       return;
    }
}

void take_and_turn()
{
    if(beepers_present())
    {
        pick_beeper();
        turn_left();
        if(!facing_west())
            take_and_turn();
        else
            take_and_turn_exit();
    }
}

void turn_east()
{
    while(!facing_east())
        turn_left();
}

void take_beepers_and_turn()
{
    turn_east();
    take_and_turn();
}

void move_to_beepers_take_turn()
{
    move_to_beepers();
    take_beepers_and_turn();
    move_to_beepers_take_turn();
}

int main()
{
    set_step_delay(100);
    turn_on("task_6.kw");
    move_to_beepers_take_turn();
    turn_off();
    return 0;
}
