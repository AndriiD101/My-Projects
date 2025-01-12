#include <superkarel.h>
void turn_right();

void turn_east();

void turn_north();

void turn_south();

void go_forward();

void wall_climb();

void build_wall();

//void climb_build();

int main()
{
    set_step_delay(100);
    turn_on("task_4.kw");
    turn_north();
    wall_climb();
    turn_off();
    return 0;
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

void go_forward()
{
    while(front_is_clear())
    {
         step();
    }
}

void wall_climb()
{
    go_forward();
    if(!front_is_clear())
    {
        if(beepers_present())
            build_wall();
        turn_south();
        while(front_is_clear())
        {
            step();
            if(beepers_present())
                build_wall();
        }
        turn_east();
        if(front_is_blocked() && right_is_blocked() && facing_east())
            turn_off();
        step();
        if(beepers_present())
            build_wall();
        turn_north();
    }
    wall_climb();
}

void build_wall()
{
    turn_north();
    go_forward();
    turn_south();
    if(!beepers_present())
        put_beeper();
    do
    {
    step();
    if(!beepers_present())
    {
            put_beeper();
    }
    }while(front_is_clear());

}
