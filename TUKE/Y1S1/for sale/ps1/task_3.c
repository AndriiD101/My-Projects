#include <superkarel.h>

void turn_right();
void turn_around();

void turn_east();

void turn_north();

void turn_south();

void turn_west();

void gather_put_first();
void gather();
void check_and_hang();

void going_back();

int main()
{
    set_step_delay(30);
    turn_on("task_3.kw");
    gather_put_first();
    check_and_hang();
    going_back();
    turn_off();
    return 0;
}

void turn_around()
{
    turn_left();
    turn_left();
}

void turn_right()
{
    turn_left();
    turn_left();
    turn_left();
}

void turn_east()
{
    while(!facing_east())
        turn_left();
}

void turn_north()
{
    while(!facing_north())
        turn_left();
}

void turn_south()
{
    while(!facing_south())
        turn_left();
}

void turn_west()
{
    while(!facing_west())
        turn_left();
}

void gather_put_first()
{
    turn_east();
    do
    {
        step();
        if(beepers_present())
            pick_beeper();
    }while(front_is_clear());
    turn_west();
    while(front_is_clear())
    {
        if(right_is_blocked())
            if(beepers_in_bag())
                put_beeper();
        step();
    }
}

void gather()
{
    do
    {
        step();
        if(beepers_present())
            pick_beeper();
    }while(front_is_clear());
    if(!beepers_in_bag())
    {
        return;
    }
}

void going_back()
{
    //set_step_delay(150);
    turn_west();
    while(front_is_clear())
        step();
    if(!front_is_clear())
        turn_north();
    while(front_is_clear())
        step();
    turn_east();
    while(!beepers_present())
    {
        if(front_is_blocked())
        {
            if(facing_east())
            {
                turn_right();
                if(front_is_blocked())
                {
                    turn_west();
                }
                else
                {
                step();
                turn_right();
                }
            }
            else if(facing_west())
            {
                turn_left();
                step();
                turn_left();
            }
        }
        step();
    }
    turn_west();
    while(front_is_clear())
        step();
    turn_east();
}

void check_and_hang()
{
    while(!front_is_clear())
    {
    turn_south();
    if(!beepers_in_bag() && front_is_blocked())
    {
        break;
    }
    step();
    turn_east();
    if(!beepers_in_bag() && front_is_blocked())
    {
        break;
    }
    gather();
    while(beepers_in_bag())
    {
    turn_north();
    step();
    if(beepers_present())
    {
        turn_south();
        step();
        if(beepers_in_bag())
            put_beeper();
    }
    else
    {
        turn_south();
        step();
    }
    turn_west();
    step();
    }
    while(front_is_clear())
        step();
    if(!beepers_in_bag() && front_is_clear())
    {
        break;
    }
    }
}


