import json

FPS = 20

def windowsToOneHot(windows):
    totalFrames = FPS * windows[0]
    v = [[0,1] for _ in range(totalFrames)]
    for i in range(1,len(windows)):
        start = int(windows[i][0] * FPS)
        end = int(windows[i][1] * FPS)
        for j in range(start,end):
            v[j] = [1,0]
    return v

#v = windowsToOneHot([20,[5,9],[14,16]])

with open('../dataset/time.json') as fp:
    obj = json.load(fp)

dataset = {}

for k in obj:
    print(k)
    dataset[k] = windowsToOneHot(obj[k])

with open('../dataset/processed/onehot.json','w') as fp:
    json.dump(dataset,fp)


