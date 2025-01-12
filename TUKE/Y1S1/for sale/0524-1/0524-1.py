import random

import matplotlib.pyplot as plt


class Worker:
    def __init__(self, start_floor, goal_floor):
        self.start_floor = start_floor
        self.goal_floor = goal_floor

    def get_start(self):
        return self.start_floor

    def get_goal(self):
        return self.goal_floor


class Elevator:
    def __init__(self, capacity, current_floor=0):
        self.capacity = capacity
        self.current_floor = current_floor
        self.moving_up = True

        self.passengers = []

    def get_capacity(self):
        return self.capacity

    def get_current_floor(self):
        return self.current_floor

    def get_passengers(self):
        return self.passengers

    def move(self):
        if self.moving_up:
            self.current_floor += 1
        else:
            self.current_floor -= 1

    def get_passenger_count(self):
        return len(self.passengers)

    def change_direction(self):
        if self.moving_up:
            self.moving_up = False
        else:
            self.moving_up = True

    def add_passenger(self, passenger):
        if self.get_passenger_count() >= self.capacity:
            return False
        else:
            self.passengers.append(passenger)
            return True

    def remove_passenger(self, passenger):
        if passenger in self.passengers:
            self.passengers.remove(passenger)
            if self.moving_up and self.get_passenger_count() == 0:
                self.change_direction()


class Building:
    def __init__(self, floor_count, elevator_count, elevator_capacity):
        self.floor_count = floor_count
        self.elevators = []
        for _ in range(elevator_count):
            self.elevators.append(
                Elevator(elevator_capacity))

        self.floors = []
        for _ in range(floor_count):
            self.floors.append([])

    def add_worker_to_floor(self, worker, floor):
        self.floors[floor].append(worker)

    def is_waiting(self):
        return len(self.floors[0])

    def time_step(self):
        for elevator in self.elevators:
            current_floor = elevator.current_floor
            for passenger in list(elevator.passengers):
                if passenger.goal_floor == current_floor:
                    elevator.remove_passenger(passenger)
            if elevator.moving_up:
                waiting_workers = []
                for worker in self.floors[current_floor]:
                    if worker not in elevator.passengers and worker.goal_floor > current_floor:
                        waiting_workers.append(worker)
                for worker in waiting_workers:
                    if elevator.get_passenger_count() < elevator.capacity:
                        elevator.add_passenger(worker)
                        self.floors[current_floor].remove(worker)
            elevator.move()
            if elevator.current_floor == 0 or elevator.current_floor == self.floor_count - 1:
                elevator.change_direction()


def simulate_workday(number_of_workers, floor_count, number_of_elevators, elevator_capacity):
    waiting_counts = []
    
    building = Building(floor_count, number_of_elevators, elevator_capacity)
    generated_workers = 0
    
    for _ in range(random.randint(1, 5)):
        worker = Worker(0, random.randint(1, floor_count-1))
        building.add_worker_to_floor(worker, 0)
        generated_workers += 1
    building.time_step()
    waiting_counts.append(len(building.floors[0]))
    
    while generated_workers <= number_of_workers or building.is_waiting():
        if generated_workers <= number_of_workers:
            for _ in range(random.randint(1, 5)):
                worker = Worker(0, random.randint(1, floor_count-1))
                building.add_worker_to_floor(worker, 0)
                generated_workers += 1
        
        building.time_step()
        waiting_counts.append(len(building.floors[0]))

    return waiting_counts


def main():
    waiting_counts = simulate_workday(1000, 10, 10, 4)
    time_step = []
    for i in range(len(waiting_counts)):
        time_step.append(i)
    
    plt.plot(time_step, waiting_counts)
    plt.show()
    # Počet čakajúcich ľudí počas príchodu do práce sa mení v závislosti od času príchodu,  pričom počas ranných špičiek je najvyšší a mimo týchto hodín sa znižuje.


if __name__ == '__main__':
    main()
