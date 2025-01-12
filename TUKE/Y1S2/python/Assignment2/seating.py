from employee import Employee
from car import Car
import itertools


def load_example(file_path):  # 2
    employees = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            driver = data[0] == 'True'
            front_seater = data[1] == 'True'
            employee = Employee(driver, front_seater)
            contacts = {}
            for contact in data[2:]:
                index, weight = contact.split(':')
                contacts[int(index)] = float(weight)
            employee.contacts = contacts
            employees.append(employee)
    return employees


def get_avg_satisfaction(employees, car_no):  # 0.5
    cars = [Car() for _ in range(car_no)]
    total_satisfaction = 0.0
    for employee in employees:
        picked_car, picked_seat = employee.take_seat(cars)
        cars[picked_car].add_passenger(employee, picked_seat)
    for car in cars:
        total_satisfaction += car.get_car_satisfaction()
    return total_satisfaction/car_no

def get_all_seatings(employees, car_no):
    drivers = [employee for employee in employees if employee.will_drive]
    non_drivers = [employee for employee in employees if not employee.will_drive]
    
    # Use combinations for the back seats
    back_seats = list(itertools.combinations(non_drivers, 2*car_no))
    
    all_seatings = []
    
    for back_seat in back_seats:
        remaining_non_drivers = list(set(non_drivers) - set(back_seat))
        
        # Use combinations for the front seats
        front_seats = list(itertools.combinations(drivers + remaining_non_drivers, car_no))
        
        for front_seat in front_seats:
            seating = []
            
            for i in range(car_no):
                # Convert the tuple to a list
                seating.append(list([front_seat[i], back_seat[2*i], back_seat[2*i+1]]))
            
            # Add the seating to all seatings
            all_seatings.append(seating)
    # print(all_seatings)
    return all_seatings


def get_optimal_satisfaction(employees, car_no):  # 0.5
    return

def get_seating_order(employees, car_no):
    orders = list(itertools.permutations(employees))

    max_satisfaction = 0
    best_order = []

    for order in orders:
        satisfaction = get_avg_satisfaction(list(order), car_no)
        if satisfaction > max_satisfaction:
            max_satisfaction = satisfaction
            best_order = order
    
    return list(best_order)