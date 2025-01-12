from employee import *


class Car:  # 1 point
    def __init__(self):  # 0.1
        self.seats = [None, None, None, None]

    def has_driver(self):  # 0.1
        return self.seats[0] != None

    def has_empty_front(self):  # 0.1
        return self.seats[1] == None

    def has_empty_back(self):  # 0.1
        return self.seats[2] == None or self.seats[3] == None

    def get_number_of_empty(self):  # 0.1
        return self.seats.count(None)

    def add_passenger(self, p, pos):  # 0.3
        if pos < 0 or pos > 3:
            raise ValueError("Cannot add passenger at index {}".format(pos))
        if self.seats[pos] is not None:
            raise ValueError("Position {} already occupied".format(pos))
        if not isinstance(p, Employee):
            raise TypeError("Cannot add non Employee variable")
        self.seats[pos] = p

    def get_car_satisfaction(self):  # 0.2
        car_satisfaction = 0
        passengers_count = 0
        for passenger in self.seats:
            if passenger is not None:
                car_satisfaction += passenger.get_satisfaction(self)
                passengers_count += 1
        return car_satisfaction / passengers_count


