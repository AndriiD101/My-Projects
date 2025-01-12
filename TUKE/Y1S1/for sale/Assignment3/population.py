import random
from person import Person
from news import *

class Population:
    def __init__(self, n, friends_count, patience_limit):
        self.people = list()
        self.active_news = list()
        self.generate_population(n, friends_count, patience_limit)

   
    def generate_population(self, n, friends_count, patience_limit):
        
        for _ in range(n):
            threshold = random.random() 
            interested_in = random.sample(CATEGORIES, 4) 
            patience = random.randint(patience_limit[0], patience_limit[1]) 
            person = Person(threshold, interested_in, patience)
            self.people.append(person)

        # Make friends
        for person in self.people:
            person.make_friends(self.people, friends_count)
       
    def introduce_news(self, news):
        interested_people = [p for p in self.people if p.is_interested_in(news.category)]
        first_five_people = interested_people[:5]
        for person in first_five_people:
            person.has_read.append(news)
        self.active_news.append(news)
        return first_five_people
    
    def update_news(self, time_step):
        self.active_news = [news for news in self.active_news if news.get_excitement(time_step) > 0]
    
    def count_readers(self, news):
        counter = 0
        for person in self.people:
            if person.has_read_news(news):
                counter += 1
        return counter

    def get_number_of_interested(self, category):
        counter = 0
        for person in self.people:
            if category in person.interested_in:
                counter += 1
        return counter


class HomogeneousPopulation(Population):
    def __init__(self, n, friends_count, patience_interval, category):
        self.category = category
        super().__init__(n, friends_count, patience_interval)

    def generate_population(self, n, friends_count, patience_limit):
        for _ in range(n):
            threshold = random.random() 
            interested_in = [self.category] + random.sample([i for i in CATEGORIES if i != self.category], 3) 
            patience = random.randint(patience_limit[0], patience_limit[1]) 
            person = Person(threshold, interested_in, patience)
            self.people.append(person)

        for person in self.people:
            person.make_friends(self.people, friends_count)
