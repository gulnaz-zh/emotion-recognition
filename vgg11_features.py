import cv2
import csv
import torchvision.transforms as transforms
import numpy as np
import os
from os import path
import torchvision.models as models

recurse = 1

root='dataset/'

net = models.vgg11(pretrained = True)    
weights = net.features[0].weight.data
w = weights.numpy()

def process(path):
    out = os.path.join(path,'VGG11activations')
    print(out)
    if (not os.path.exists(out)):
        os.mkdir(out)
    result = []
#    cnt = 0
    files = os.listdir(path) 
#    for root, subdirs, files in os.walk(rootdir):
    print(len(files))
    with open(out+'/filenames.csv', mode='w') as filecsv:
        writer = csv.writer(filecsv)
        writer.writerow(files)
    for filename in os.listdir(path + "/"):
        img = cv2.imread(os.path.join(path + "/",filename))
        if img is not None:
            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = transforms.ToTensor()(img)
            transform = transform.reshape(1, transform.shape[0], transform.shape[1], transform.shape[2])
            out1 = net.features[0](transform)
            out2 = net.features[1](out1)
            out3 = net.features[2](out2)
            out4 = net.features[3](out3)
            out5 = net.features[4](out4)
            out6 = net.features[5](out5)
            out7 = net.features[6](out6)
            out8 = net.features[7](out7)
            out9 = net.features[8](out8)
            out10 = net.features[9](out9)
            out11 = net.features[10](out10)
            out12 = net.features[11](out11)
            out13 = net.features[12](out12)
            out14 = net.features[13](out13)
            out15 = net.features[14](out14)
            out16 = net.features[15](out15)
            out17 = net.features[16](out16)
            out18 = net.features[17](out17)
            out19 = net.features[18](out18)
            out20 = net.features[19](out19)
            out21 = net.features[20](out20)

            list = []
            for i in range(0, 64):
                output = out1.detach().numpy()
                list.append( np.mean(output[0][i]) )

            for i in range(0, 128):
                output = out4.detach().numpy()
                list.append( np.mean(output[0][i]) )

            for i in range(0, 256):
                output = out7.detach().numpy()
                list.append( np.mean(output[0][i]) )

            for i in range(0, 256):
                output = out9.detach().numpy()
                list.append( np.mean(output[0][i]) )

            for i in range(0, 512):
                output = out12.detach().numpy()
                list.append( np.mean(output[0][i]) )

            for i in range(0, 512):
                output = out14.detach().numpy()
                list.append( np.mean(output[0][i]) )

            for i in range(0, 512):
                output = out17.detach().numpy()
                list.append( np.mean(output[0][i]) )

            for i in range(0, 512):
                output = out19.detach().numpy()
                list.append( np.mean(output[0][i]) )


            vector = np.array(list)
            list.clear()
            result.append(vector)
    final = np.array(result)
    print(final)
   # print(len(final))
    np.savetxt(out+'/VGG11_averages.csv', final, delimiter=',')


if (recurse == 1):
    for folder in os.listdir(root):
        print(os.path.join(root,folder))
        process(os.path.join(root,folder))
    else:
        process(root)

#    classAverage = np.mean(final, axis = 0)
#    np.savetxt('./'+folder+'/class_average_by_column.csv', classAverage, delimiter=',')
#result.clear()



