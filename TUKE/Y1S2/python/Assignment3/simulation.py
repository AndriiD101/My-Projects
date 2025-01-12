from news import *
from person import *
from population import *
import numpy as np

def simulate_spread(all_news, population):
    readers_dict = {news: [] for news in all_news}
    initial_readers = {news: population.introduce_news(news) for news in all_news}
    time_step = 2

    while population.active_news:
        for news in all_news:
            readers = initial_readers[news]
            new_readers = []
            for reader in readers:
                forwarded_to = reader.process_news(news, time_step)
                new_readers.extend(forwarded_to)
            initial_readers[news] = new_readers
            reader_count = population.count_readers(news)
            readers_dict[news].append(reader_count)
        population.update_news(time_step)
        time_step += 1
    max_length = max(len(readers) for readers in readers_dict.values())
    for news, readers in readers_dict.items():
        if len(readers) < max_length:
            readers += [readers[-1]] * (max_length - len(readers))

    return readers_dict

def average_spread_with_excitement_rate(excitement_rate, pop_size, friends_count, patience_limit, test_count=1):
    return None, None

def excitement_to_reach_percentage(percentage, pop_size, friends_count, patience_limit):
    return

def excitement_to_reach_percentage_special_interest(
        percentage, pop_size, friends_count, patience_limit, news_category):
    return


if __name__ == '__main__':
    pass
