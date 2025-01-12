import random
import matplotlib.pyplot as plt

class Table:
    def __init__(self, capacity):
        self.capacity = capacity
        self.seated_people = 0
    
    def seat_people(self, people_number):
        if people_number > self.capacity:
            raise ValueError("Group size exceeds table capacity")
        self.seated_people += people_number
    
    def remove_group(self):
        self.seated_people = 0

class Group:
    def __init__(self, size, stays_for, arrived_at):
        self.size = size
        self.stays_for = stays_for
        self.arrived_at = arrived_at
        self.table = None
    
    def arrive(self, time_step, table):
        self.table = table
    
    def should_leave(self, time_step):
        stayed_for = time_step - self.arrived_at
        return stayed_for >= self.stays_for
    
    def leave(self):
        if self.table:
            self.table.remove_group()
        self.table = None

class Restaurant:
    def __init__(self, tables):
        self.tables = tables
        self.groups = []
    
    def add_group(self, group, time_step):
        available_tables = []
        for table in self.tables:
            if table.seated_people + group.size <= table.capacity:
                available_tables.append(table)
        
        if available_tables:
            chosen_table = random.choice(available_tables)
            chosen_table.seat_people(group.size)
            group.arrive(time_step, chosen_table)
            self.groups.append(group)
    
    def simulate_step(self, time_step):
        groups_to_remove = []
        for group in self.groups:
            if group.should_leave(time_step):
                group.leave()
                groups_to_remove.append(group)
        for group in groups_to_remove:
            self.groups.remove(group)
    
    def get_rate_of_use(self):
        total_seats = sum(table.capacity for table in self.tables)
        occupied_seats = sum(table.seated_people for table in self.tables)
        return (occupied_seats / total_seats) * 100
    
    def simulate(self, length, table_count, group_chance):
        rates = []
        for _ in range(table_count):
            capacity = random.choice([2, 4])
            self.tables.append(Table(capacity))

        for time_step in range(length):
            if random.random() < group_chance:
                group_size = random.randint(1, 4)
                group_stays_for = random.randint(20, 40)
                new_group = Group(group_size, group_stays_for, time_step)
                self.add_group(new_group, time_step)
            self.simulate_step(time_step)
            rates.append(self.get_rate_of_use())

        return rates

def main():
    length = 100
    table_count = 10
    group_chance = 0.3

    restaurant = Restaurant([])
    rates = restaurant.simulate(length, table_count, group_chance)

    plt.plot(range(length), rates)
    plt.title("Seat Utilization Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Seat Utilization (%)")
    plt.show()

if __name__ == "__main__":
    main()
