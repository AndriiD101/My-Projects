#include <superkarel.h>

void fill();

void go_to_right_corner();

void horizontal_middle();
void vertical_middle();

void turn_around();

void FacingNorth();
void FacingSouth();
void FacingWest();
void FacingEast();

int main()
{
    set_step_delay(70);
    turn_on("task_5.kw");
    fill();
    horizontal_middle();
    vertical_middle();
    turn_off();
    return 0;
}

void FacingSouth()
{
    while(!facing_south())
        turn_left();
}

void FacingWest()
{
    while(!facing_west())
        turn_left();
}

void FacingEast()
{
    while(!facing_east())
        turn_left();
}


void FacingNorth()
{
    while(!facing_north())
        turn_left();
}

void vertical_middle()
{
    do
    {
        step();
    }while(!beepers_present() && front_is_clear());
    FacingSouth();
    step();
    if(!beepers_present())
        put_beeper();
    do
    {
        step();
    }while(!beepers_present() && front_is_clear());
    FacingNorth();
    step();
    if(!beepers_present())
        put_beeper();

    while(beepers_present())
    {
        do {
            step();
        } while (!beepers_present());
        if(front_is_clear())
            pick_beeper();
        turn_around();
        step();
        if (no_beepers_present() && front_is_clear()) {
            put_beeper();
        } else {
            while (beepers_present()) {
                pick_beeper();
            }
        }
    }
    FacingNorth();

}


//void horizontal_middle()
//{
//    go_to_right_corner();
//    FacingNorth();
//    step();
//    FacingWest();
    //do
    //{
    //   step();
    //}while(!beepers_present() &&front_is_clear());
    //FacingEast();
    //step();
    //if(!beepers_present())
    //    put_beeper();
    //do
    //{
    //    step();
    //}while(!beepers_present() && front_is_clear());
    //FacingWest();
    //do
    //{
    //    step();
    //if(!beepers_present())
    //    put_beeper();
    //step();
    //if(beepers_present())
    //    pick_beeper();
    //step();
    //if(beepers_present())
    //{
    //    pick_beeper();
    //    FacingNorth();
    //    break;
    //}
    //put_beeper();
    //do
    //    step();
    //while(!beepers_present());
    //if(!beepers_present())
    //   pick_beeper();
    //turn_around();
    //}while(!beepers_present() && front_is_clear());


//}

void horizontal_middle()
{
    do
    {
        step();
    }while(!beepers_present() && front_is_clear());
    FacingWest();
    step();
    if(!beepers_present())
        put_beeper();
    do
    {
        step();
    }while(!beepers_present() && front_is_clear());
    FacingEast();
    step();
    if(!beepers_present())
        put_beeper();
    while(beepers_present())
    {
        do {
            step();
        } while (!beepers_present());
        if(front_is_clear())
            pick_beeper();
        turn_around();
        step();
        if (no_beepers_present() && front_is_clear()) {
            put_beeper();
        } else {
            while (beepers_present()) {
                pick_beeper();
            }
        }
    }
    FacingNorth();

}


void turn_around()
{
    turn_left();
    turn_left();
}


void go_to_right_corner()
{
    FacingEast();
    while(front_is_clear())
        step();
    FacingSouth();
    while(front_is_clear())
        step();
}

void fill()
{
    go_to_right_corner();
    FacingNorth();
    while(!beepers_present())
    {
        put_beeper();
        step();
        if(!front_is_clear())
            turn_left();
    }
}
