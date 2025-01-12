import pickle
with open('points.pkl', 'rb') as f:
    data = pickle.load(f)
    print (data)