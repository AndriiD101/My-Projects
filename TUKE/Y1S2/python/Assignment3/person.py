import random

from news import News

class Person:
    def __init__(self, threshold, interested_in, patience):
        self.threshold = threshold
        self.interested_in = interested_in
        self.friends_list = list()
        self.has_read = list()
        self.patience = patience

    def is_interested_in(self, category):
        return category in self.interested_in

    def has_read_news(self, news):
        return news in self.has_read

    def make_friends(self, population, n):
            # Remove the current person from the population
            population_without_self = [person for person in population if person != self]

        # Make friends
            self.friends_list = random.sample(population_without_self, min(n, len(population_without_self)))

    def process_news(self, news, time_step):  # 1b
        forward_to = []
        # if len(self.has_read)>self.patience:
        #     return []
        if news in self.has_read:
            return []
        if news.category not in self.interested_in:
            return []
        if news.get_excitement(time_step) <= self.threshold:
            return []
        
        for friend in self.friends_list:
            if news.category in friend.interested_in:
                forward_to.append(friend)
        
        self.has_read.append(news)

        return forward_to