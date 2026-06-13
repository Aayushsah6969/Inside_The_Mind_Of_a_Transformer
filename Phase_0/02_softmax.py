import math

y = [1000, 1001, 999]
z = max(y)
x=[]
for i in y:
    x.append(i-z)

list = []
softs = []

def softmax(x):
    for i in x:
        print(i)
        print("exp(i) = ", math.exp(i))
        list.append(math.exp(i))
        print(list)

    sum = 0
    for i in list:
        sum += i

    print("sum = ", sum)

    for i in list:
        softs.append(i/sum)

    print("Softmax values = ", softs)

    prob_sum = 0
    for j in softs:
        prob_sum += j

    print("Probability sum = ", prob_sum)


softmax(x)


##### APPROACH WITH TORCH #####
import torch
import torch.nn.functional as F

scores = torch.tensor(x, dtype=torch.float)

print(F.softmax(scores, dim=0))