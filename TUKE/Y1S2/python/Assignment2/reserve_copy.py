class Employee:  # 3 points

    def __init__(self, driver, front_seater=True):  # 0.25
        self.contacts = {}
        self.will_drive:bool = driver
        self.sits_in_front = front_seater

    def set_contacts(self, contacts):  # 0.25
        for name, value in contacts:
            if not isinstance(name, Employee):
                raise TypeError("first value in contacts must be Employee object")
            if name in self.contacts:
                raise ValueError("This name is already in contact")
            self.contacts[name] = value

    def get_car_weight(self, car):  # 0.5
        total_car_value = 0
        for passenger in car.seats:
            if passenger in self.contacts:
                total_car_value += self.contacts[passenger]
        return total_car_value
    
    def get_satisfaction(self, car):  # 0.5
        if self not in car.seats:
            return -1
        car_value = self.get_car_weight(car)
        satisfaction_constant = 0.5
        seat_index = car.seats.index(self)
        if (self.will_drive and self.sits_in_front and seat_index == 0) or (not self.will_drive and not self.sits_in_front and seat_index in [2, 3]) or (self.sits_in_front and not self.will_drive and seat_index in [0, 1]):
            satisfaction_constant = 2
        return car_value * satisfaction_constant

    def choose_car(self, cars):  # 0.5
        cars_with_free_seat = []
        for car in cars:
            if None in car.seats:
                cars_with_free_seat.append(car)
        cars_with_free_seat.sort(key=self.get_car_weight, reverse=True)

        for car in cars_with_free_seat:
            if (not self.will_drive and self.sits_in_front) and car.has_empty_front():
                return car
            if (self.will_drive and self.sits_in_front) and not car.has_driver():
                return car
            if (self.will_drive and self.sits_in_front) and not car.has_driver():
                return car
            if not self.will_drive and not self.sits_in_front and car.has_empty_back():
                return car
        return cars_with_free_seat[0]

    def take_seat(self, cars):  # 1
        picked_car = self.choose_car(cars)
        if self.will_drive and not picked_car.has_driver():
            picked_seat = 0
        elif self.sits_in_front and picked_car.has_empty_front():
            picked_seat = 1
        elif not self.will_drive and not self.sits_in_front and picked_car.has_empty_back():
            picked_seat = 2 if picked_car.seats[2] is None else 3
        else:
            for seat_index, seat in enumerate(picked_car.seats):
                if seat is None:
                    picked_seat = seat_index
                    break
        return cars.index(picked_car), picked_seat
