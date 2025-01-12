def count_day_and_night_shifts_from_file(filename):
    try:
        with open(filename, 'r') as file:
            schedule = [list(map(int, line.split())) for line in file if line.strip()]

        max_member = max(max(day) for day in schedule)

        day_shifts = {i: 0 for i in range(1, max_member + 1)}
        night_shifts = {i: 0 for i in range(1, max_member + 1)}

        for day in schedule:
            day_shifts[day[0]] += 1

            for night_worker in day[1:]:
                night_shifts[night_worker] += 1

        for member in range(1, max_member + 1):
            print(f"Працівник {member}: Денних змін: {day_shifts[member]}, Нічних змін: {night_shifts[member]}")

    except FileNotFoundError:
        print(f"Файл {filename} не знайдений.")
    except Exception as e:
        print(f"Сталася помилка: {e}")

filename = 'schedule.txt'
count_day_and_night_shifts_from_file(filename)
