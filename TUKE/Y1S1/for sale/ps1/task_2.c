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

void go_through_maze()
{
    if(right_is_blocked() && front_is_blocked())
        turn_left();
}

void go_back()
{
    if(left_is_blocked() && front_is_blocked())
        turn_right();
}



int main()
{
    set_step_delay(100);
    turn_on("task_2.kw");
    while(no_beepers_in_bag())
    {
        go_through_maze();
        step();
        if(beepers_present())
            pick_beeper();
    }
    turn_around();
    while(true)
    {
        go_back();
        step();
        if(front_is_blocked() && right_is_blocked() && left_is_blocked())
            break;
    }
    while(!facing_north())
        turn_left();

    turn_off();
    return 0;

}

