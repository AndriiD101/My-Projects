CATEGORIES = ["politics", "world", "culture", "tech", "local", "sport"]


class News:
    def __init__(self, category, excitement_rate, validity_length, created):
        self.check_data(category, excitement_rate, validity_length, created)

        self.category = category
        self.excitement_rate = excitement_rate
        self.validity_length = validity_length
        self.created = created

    def check_data(self, category, excitement_rate, validity_length, created):
        if category not in CATEGORIES: 
            raise ValueError
        if not isinstance(excitement_rate, float):
            raise TypeError
        if excitement_rate<0 or excitement_rate>1:
            raise ValueError
        if not isinstance(validity_length, int):
            raise TypeError
        if validity_length<1 or validity_length>10:
            raise ValueError
        if not isinstance(created, int):
            raise TypeError
        if created<1:
            raise ValueError

    def get_excitement(self, time_step):
        if time_step>self.validity_length+self.created:
            return 0.0
        return self.excitement_rate**(time_step-self.created)

