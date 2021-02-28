import random
print(random.sample([[i,j] for i in range(15) for j in range(15)],random.randint(1,3)))