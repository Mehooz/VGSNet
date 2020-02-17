import torch
from model import *
class PSGNet(nn.Module):
    def __init__(self,output_points=1040):
        super(PSGNet,self).__init__()
        self.conv2d_input  = conv_2d(3,16,3,padding=1,batch_norm=False)
        self.conv2d_16     = conv_2d(16,16,3,padding=1,batch_norm=False)
        self.conv2d_16_3   = conv_2d(16, 16, 3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_16_5   = conv_2d(32,16,3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_32     = conv_2d(32,32,3,padding=1,batch_norm=False)
        self.conv2d_32_2   = conv_2d(16,32,3,2,padding=1,batch_norm=False)
        self.conv2d_32_3   = conv_2d(32, 32, 3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_32_4   = conv_2d(16, 32, 3,2 , padding=1,activation='linear',batch_norm=False)
        self.conv2d_32_5   = conv_2d(64,32,3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_64     = conv_2d(64,64,3,padding=1,batch_norm=False)
        self.conv2d_64_2   = conv_2d(32,64,3,2,padding=1,batch_norm=False)
        self.conv2d_64_3   = conv_2d(64, 64, 3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_64_4   = conv_2d(32, 64, 3,2, padding=1,activation='linear',batch_norm=False)
        self.conv2d_64_5   = conv_2d(128,64,3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_128    = conv_2d(128,128,3,padding=1,batch_norm=False)
        self.conv2d_128_2  = conv_2d(64,128,3,2,padding=1,batch_norm=False)
        self.conv2d_128_3  = conv_2d(128, 128, 3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_128_4  = conv_2d(64, 128, 3, 2,padding=1,activation='linear',batch_norm=False)
        self.conv2d_128_5  = conv_2d(256,128,3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_256    = conv_2d(256,256,3,padding=1,batch_norm=False)
        self.conv2d_256_2  = conv_2d(128,256,5,2,padding=2,batch_norm=False)
        self.conv2d_256_3  = conv_2d(256,256,3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_256_4  = conv_2d(128,256,3,2,padding=1,activation='linear',batch_norm=False)
        self.conv2d_256_5  = conv_2d(512,256,3,padding=1,activation='linear',batch_norm=False)
        self.conv2d_512    = conv_2d(512,512,3,padding=1,batch_norm=False)
        self.conv2d_512_2  = conv_2d(256,512,5,2,padding=2,batch_norm=False)
        self.conv2d_512_3  = conv_2d(512,512,5,2,padding=2,batch_norm=False)
        self.conv2d_3 = conv_2d(64,3,3,padding=1,activation='linear',batch_norm=False)


        self.fc_layer_1 = fc_layer(512*16,2048,batch_norm=False)
        self.fc_layer_2 = fc_layer(2048,1024,batch_norm=False)
        self.fc_layer_3 = fc_layer(1024,256*3,batch_norm=False)
        self.fc_layer_4 = fc_layer(2048,2048,batch_norm=False)


        self.conv2d_transpose_512 = conv_2d_transpose(512,256,5,2,padding=2,activation='linear',batch_norm=False)
        self.conv2d_transpose_256 = conv_2d_transpose(256,128,5,2,padding=2,output_padding=1,activation='linear',batch_norm=False)
        self.conv2d_transpose_128 = conv_2d_transpose(128,64,5,2,padding=2,output_padding=1,activation='linear',batch_norm=False)
        self.conv2d_transpose_64 = conv_2d_transpose(64,32,5,2,padding=2,output_padding=1,activation='linear',batch_norm=False)
        self.conv2d_transpose_32 = conv_2d_transpose(32,16,5,2,padding=2,output_padding=1,activation='linear',batch_norm=False)

        self.relu = relu_layer()
        self.output_points= output_points


    def forward(self, x):

        batch_size = x.size(0)
        # 192 256
        x = self.conv2d_input(x)
        x = self.conv2d_16(x)
        x = self.conv2d_16(x)
        x0 =x
        x = self.conv2d_32_2(x)

        # 96 128
        x = self.conv2d_32(x)
        x = self.conv2d_32(x)
        x1 = x
        x = self.conv2d_64_2(x)

        # 48 64
        x = self.conv2d_64(x)
        x = self.conv2d_64(x)
        x2 = x
        x = self.conv2d_128_2(x)

        # 24 32
        x = self.conv2d_128(x)
        x = self.conv2d_128(x)
        x3 = x
        x = self.conv2d_256_2(x)

        # 12 16
        x = self.conv2d_256(x)
        x = self.conv2d_256(x)
        x4 = x
        x = self.conv2d_512_2(x)

        # 6 8
        x = self.conv2d_512(x)
        x = self.conv2d_512(x)
        x = self.conv2d_512(x)
        x5 = x
        x = self.conv2d_512_3(x)

        # 3 4
        x_additional = self.fc_layer_1(x.view(batch_size,-1))
        x = self.conv2d_transpose_512(x)

        # 6 8
        x5 = self.conv2d_256_5(x5)
        x = self.relu(x.add(x5))
        x = self.conv2d_256(x)
        x5 = x
        x = self.conv2d_transpose_256(x)

        # 12 16
        x4 = self.conv2d_128_5(x4)
        x = self.relu(x.add(x4))
        x = self.conv2d_128(x)
        x4 = x
        x = self.conv2d_transpose_128(x)

        # 24 32
        x3 = self.conv2d_64_5(x3)
        x = self.relu(x.add(x3))
        x = self.conv2d_64(x)
        x3 = x
        x = self.conv2d_transpose_64(x)

        # 48 64
        x2 = self.conv2d_32_5(x2)
        x = self.relu(x.add(x2))
        x = self.conv2d_32(x)
        x2 = x
        x = self.conv2d_transpose_32(x)

        # 96 128
        x1 = self.conv2d_16_5(x1)
        x = self.relu(x.add(x1))
        x = self.conv2d_16(x)
        x = self.conv2d_32_4(x)

        # 48 64
        x2 = self.conv2d_32_3(x2)
        x = self.relu(x.add(x2))
        x = self.conv2d_32(x)
        x2 = x
        x = self.conv2d_64_4(x)

        # 24 32
        x3 = self.conv2d_64_3(x3)
        x = self.relu(x.add(x3))
        x = self.conv2d_64(x)
        x3 = x
        x = self.conv2d_128_4(x)

        # 12 16
        x4 = self.conv2d_128_3(x4)
        x = self.relu(x.add(x4))
        x = self.conv2d_128(x)
        x4 = x
        x = self.conv2d_256_4(x)

        # 6 8
        x5 = self.conv2d_256_3(x5)
        x = self.relu(x.add(x5))
        x = self.conv2d_256(x)
        x5 = x
        x = self.conv2d_512_2(x)

        # 3 4
        x_additional = self.fc_layer_4(x_additional)
        x_additional = self.relu(x_additional.add(self.fc_layer_1(x.view(batch_size,-1))))
        x = self.conv2d_transpose_512(x)

        # 6 8
        x5 = self.conv2d_256_3(x)
        x = self.relu(x.add(x5))
        x = self.conv2d_256(x)
        x5 = x
        x = self.conv2d_transpose_256(x)

        # 12 16
        x4 = self.conv2d_128_3(x)
        x = self.relu(x.add(x4))
        x = self.conv2d_128(x)
        x4 = x
        x = self.conv2d_transpose_128(x)

        # 24 32
        x3 = self.conv2d_64_3(x)
        x = self.relu(x.add(x3))
        x = self.conv2d_64(x)
        x3 = x
        x = self.conv2d_64(x)

        x_additional = self.fc_layer_2(x_additional)
        x_additional = self.fc_layer_3(x_additional)
        x_additional = x_additional.view([batch_size,256,3])

        x =self.conv2d_3(x)
        x = x.view(batch_size,28*28,3)
        x_feature = x
        x = torch.cat([x_additional,x],1)
        x = x.view(batch_size, self.output_points, 3)

        return x,x_feature


if __name__ == '__main__':
    net = PSGNet(3000)
    net = net.to(device='cuda')
    net = nn.DataParallel(net)
    net.train()
    img = torch.rand([32,3,224,224])
    net(img)
