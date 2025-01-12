#include <superkarel.h>

void turn_right();

void turn_east();

void turn_north();

void turn_south();

void turn_west();

void put_line_of_beepers();

void find_inapropriate();

int main()
{
    set_step_delay(100);
    turn_on("task_7.kw");
    put_line_of_beepers();
    find_inapropriate();
    turn_off();
    return 0;
}

void turn_right()
{
    turn_left();
    turn_left();
    turn_left();
}

void turn_west()
{
    while(!facing_west())
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

void put_line_of_beepers()
{
    turn_east();
    if(left_is_clear())
    {
        put_beeper();
    }
    while(front_is_clear())
    {
        step();
        if(left_is_clear())
        {
            put_beeper();
        }
    }
    turn_west();
    while(front_is_clear())
        step();
}

void take_all_back()
{
    turn_west();
    while(front_is_clear())
        step();
    turn_east();
    while(front_is_clear())
        step();
}

void find_inapropriate()
{
    turn_east();
    while(!beepers_present())
    {
        step();
        if(front_is_blocked())
            return;
    }
    if(left_is_clear())
    {
        turn_left();
        pick_beeper();
        step();
        if(beepers_present())
        {
            turn_south();
            step();
            turn_east();
            do
                step();
            while(!beepers_present());
        }

        else if(!beepers_present())
        {
            if(!beepers_present())
                put_beeper();
            do {
                if(!left_is_blocked())
                turn_left();
                while(front_is_blocked() && left_is_blocked())
                    turn_right();
                if(front_is_blocked())
                    return;
                step();
            } while(!beepers_present());
        if(beepers_present() && front_is_blocked())
        {
            pick_beeper();
            turn_north();
            step();
            if(beepers_present())
                pick_beeper();
            else
                put_beeper();
            turn_south();
            step();
            turn_west();
            while(front_is_clear())
                step();
        }
        else if(beepers_present() && front_is_clear())
        {
            pick_beeper();
            step();
        }
        }
    turn_west();
    while(beepers_present())
        step();
    turn_east();
    find_inapropriate();
}
}
