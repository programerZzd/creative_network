import argparse
import os
import random
from statistics import mean

from PIL import Image
from einops import rearrange

from clip import clip
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, autocast
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
best_prompt = ["African crocodile&Irish setter",
                    "banded gecko&indigo bunting",
                    "cowboy hat&half track",
                    "hippopotamus&fireboat",
                    "lawn mower&half track",
                    "lionfish&abacus",
                    "oboe&jellyfish",
                    "oranges&fox",
                    "orangutan&king crab",
                    "orchids&possum",
                    "overskirt&ibex",
                    "raccoon&mushrooms",
                    "roses&raccoon",
                    "vestment&broccoli",
                    "wild boar&cash machine",
                    "Windsor tie&wolf spider",
                    "American lobster&rock python",
                    "oxcart&crash helmet",
                    "pineapple&bagel",
                    "Polaroid camera&electric locomotive",
                    "punching bag&amphibian",
                    "sax&pirate",
                    "toilet seat&tow truck",
                    "agaric&lumbermill",
                    "barrow&cucumber",
                    "bee eater&china cabinet",
                    "cassette&jigsaw puzzle",
                    "croquet ball&warplane",
                    "crane&puck",
                    "ear&cardoon",
                    "macaque&timber wolf",
                    "plane&fly",
                    "toilet tissue&drum",
                    "pomegranate&teapot",
                    ]

animal_class = ['bird'
        'reptile'
        'fish'
        'invertebrate'
'mammal'
'fungus'
'body_part']

def draw(losses, dir):
    plt.plot(range(len(losses)), losses, c='r')
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.savefig(dir)


def inter_str_convert(l):
    l_meshenter = l[2: -2].split("\n ")
    l_lst = list()
    for l_m in l_meshenter:
        if ". " in l_m:
            l_lst += l_m.split(". ")
        else:
            l_lst += l_m.split(" ")
    l_lst = list(filter(lambda x: x != '', l_lst))
    size = len(l_lst)
    T = torch.full([1, size], 0, dtype=torch.float32)
    # T = T - To
    for pos, i in zip(l_lst, range(size)):
        if '1' in pos:
            T[0, i] = 1
    return T


def genRandomT_interpolate(size, shuffle):
    '''

    :param size: total size
    :param part_size: size of perm
    :return:
    '''
    T = np.zeros([1, size])
    posT = random.sample(range(int(size)), int(int(size) * shuffle))
    for i in posT:
        # print(i, posT[i])
        T[0, i] = 1
    # resT = np.array([T, T, T, T])
    return torch.tensor(T).type(torch.float32)


class BSNet(nn.Module):
    # 1*768 -> 77*768 -> 1*768
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(154, 32)
        self.fc2 = nn.Linear(32, 10)
        # self.bn1 = nn.BatchNorm2d()
        self.fc3 = nn.Linear(10, 2)
        self.fc4 = nn.Linear(16, 5)
        self.fc5 = nn.Linear(5, 1)
        self.lsm = nn.Softmax(dim=0)

    def forward(self, T):
        # print(T.shape)
        T = T.view(1, -1)
        # T = self.bn1()
        T = torch.relu(self.fc1(T))
        T = torch.relu(self.fc2(T))
        T = torch.sigmoid(self.fc3(T))

        # T = F.softmax(T, dim=0)
        # print(T)
        return T


class SNet(nn.Module):
    # 1*768 -> 77*768 -> 1*768
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(8, 1, 3, 3)
        self.cnn2 = nn.Conv2d(1, 1, 3, 2)
        self.fc1 = nn.Linear(12 * 170, 1024)
        self.flatten = nn.Flatten(start_dim=1)
        # self.fc2 = nn.Linear(77*1024, 10*1024)
        # self.fc3 = nn.Linear(10*1024, 1024)
        # self.bn1 = nn.BatchNorm2d()

    def forward(self, T):
        # print(T.shape)
        #
        T = torch.tanh(self.cnn1(T))
        T = torch.tanh(self.cnn2(T))
        # T = T.view(1, -1)
        T = self.flatten(T)
        # print(T.shape)
        # T = self.bn1()
        T = torch.tanh(self.fc1(T))
        # T = torch.relu(self.fc2(T))
        # T = self.fc3(T)

        # T = F.softmax(T, dim=0)
        # print(T)
        return T


class SNet_deep(nn.Module):
    # 1*768 -> 77*768 -> 1*768
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(8, 4, 3, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.cnn2 = nn.Conv2d(4, 1, 3, 1)
        self.bn2 = nn.BatchNorm2d(1)
        self.cnn3 = nn.Conv2d(1, 1, 3, 3)

        self.fc1 = nn.Linear(24 * 340, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.drop = nn.Dropout(0.2)
        self.flatten = nn.Flatten(start_dim=1)
        # self.fc2 = nn.Linear(77*1024, 10*1024)
        # self.fc3 = nn.Linear(10*1024, 1024)
        # self.bn1 = nn.BatchNorm2d()

    def forward(self, T):
        # print(T.shape)
        #
        T = torch.tanh(self.bn1(self.cnn1(T)))
        T = torch.tanh(self.bn2(self.cnn2(T)))
        T = torch.tanh(self.cnn3(T))
        # print(T.shape)
        # T = T.view(1, -1)
        T = self.flatten(T)
        # print(T.shape)
        # T = self.bn1()
        T = torch.tanh(self.fc1(T))
        T = self.drop(T)
        T = torch.tanh(self.fc2(T))
        # T = torch.relu(self.fc2(T))
        # T = self.fc3(T)

        # T = F.softmax(T, dim=0)
        # print(T)
        return T

class SNet(nn.Module):
    # 1*768 -> 77*768 -> 1*768
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(8, 1, 3, 3)
        self.cnn2 = nn.Conv2d(1, 1, 3, 2)

        self.fc1 = nn.Linear(12 * 170, 1024)
        self.flatten = nn.Flatten(start_dim=1)
        # self.fc2 = nn.Linear(77*1024, 10*1024)
        # self.fc3 = nn.Linear(10*1024, 1024)
        # self.bn1 = nn.BatchNorm2d()

    def forward(self, T):
        # print(T.shape)
        #
        T = torch.tanh(self.cnn1(T))
        T = torch.tanh(self.cnn2(T))
        # print(T.shape)
        # T = T.view(1, -1)
        T = self.flatten(T)
        # print(T.shape)
        # T = self.bn1()
        T = torch.tanh(self.fc1(T))
        # T = torch.relu(self.fc2(T))
        # T = self.fc3(T)

        # T = F.softmax(T, dim=0)
        # print(T)
        return T

class SNet5(nn.Module):
    # 1*768 -> 77*768 -> 1*768
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(8, 1, 3, 3)
        self.cnn2 = nn.Conv2d(1, 1, 3, 2)
        self.fc1 = nn.Linear(12 * 170, 1024)
        self.flatten = nn.Flatten(start_dim=1)
        # self.fc2 = nn.Linear(77*1024, 10*1024)
        # self.fc3 = nn.Linear(10*1024, 1024)
        # self.bn1 = nn.BatchNorm2d()

    def forward(self, T):
        # print(T.shape)
        #
        T = torch.tanh(self.cnn1(T))
        T = torch.tanh(self.cnn2(T))
        # T = T.view(1, -1)
        T = self.flatten(T)
        # print(T.shape)
        # T = self.bn1()
        T = torch.tanh(self.fc1(T))
        # T = torch.relu(self.fc2(T))
        # T = self.fc3(T)

        # T = F.softmax(T, dim=0)
        # print(T)
        return T


def make_dataset(frac):
    dir = "./my_eval_check/res_file"
    weight_dir = "/opt/data/private/stable_diffusion/my_origin_data_inter"
    prompts_list = os.listdir(dir)
    outname = "interset"
    dataset_dict = {}
    inter_list = list()
    file_list = list()
    for prompts in tqdm(prompts_list):
        file = int(os.listdir(os.path.join(dir, prompts))[0].split("_")[0])
        df = pd.read_csv(os.path.join(weight_dir, prompts, "weight", prompts + ".csv"))
        inter_str = df.loc[df["img_name"] == file+1, "interpolate_position"].values[0]
        inter = inter_str_convert(inter_str).numpy()
        inter_list.append(inter)
        file_list.append(os.listdir(os.path.join(dir, prompts))[0])

    df = pd.DataFrame(data=list(zip(prompts_list, file_list, inter_list)), columns=["prompt","file","inters"])
    df.to_csv("df_results/inters/all_best.csv",index=False)
    if frac!= 1:
        df_f = df.sample(frac=frac).reset_index(drop=True)
        df_f.to_csv("df_results/inters/all_best_demo.csv", index=False)
    df_train = df.sample(frac=0.8).reset_index(drop=True)
    df_test = df.sample(frac=0.2).reset_index(drop=True)
    os.makedirs("df_results/" + outname, exist_ok=True)
    print(df_train.shape)
    df_train.to_csv("df_results/" + outname + "/train.csv", index=False)
    df_test.to_csv('df_results/' + outname + '/test.csv', index=False)


def load_perm(dir):
    df = pd.read_csv(dir)
    prompts = df['prompt'].tolist()
    poses = df['inter'].tolist()
    return prompts, poses


def genInter_fromList(l):
    l = l[1: -1].split(", ")
    size = len(l)
    T = torch.full([1, size], -1)
    # T = T - To
    for pos, i in zip(l, range(size)):
        if pos == '1.0':
            T[0, i] = 1
    return T


class my_inter_set_dis(Dataset):
    def __init__(self, train=True, device="cuda:0"):
        self.dir = "/opt/data/private/stable_diffusion/my_origin_data_inter"
        self.prompts_list = os.listdir(self.dir)
        if train:
            self.prompts = random.sample(self.prompts_list, int(0.8 * len(self.prompts_list)))
        else:
            self.prompts = random.sample(self.prompts_list, int(0.2 * len(self.prompts_list)))
        self.device = device

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompts = self.prompts[index]
        dir = self.dir
        c = torch.load(os.path.join(dir, prompts, "weight", prompts + ".t")).to(self.device)
        inter = torch.load(os.path.join(dir, prompts, "weight", prompts + "_count.t")).to(self.device)
        # print(inter)
        return prompts, c, inter[0], inter[1]


class my_inter_set(Dataset):
    def __init__(self, dir, device="cuda:0"):
        self.df = pd.read_csv(dir)
        self.datadir = "/opt/data/private/stable_diffusion/my_origin_data_inter"
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        s = self.df.iloc[index]
        inter = inter_str_convert(s["inters"])
        prompts = s["prompt"]
        c = torch.load(os.path.join(self.datadir, prompts, "weight", prompts + ".t")).to(self.device)
        # print(c.type())
        # print(inter.type())
        # inter = torch.load(os.path.join(self.datadir, prompts, "weight", prompts + "_best_inter.t")).to(
        #     torch.float32).to(self.device)
        # print(inter)
        return prompts, c, inter[0]

class my_inter_set_style(Dataset):
    def __init__(self, dir, device="cuda:0", difmodel=None):
        self.styles = ["Aquarell",
                        "line art",
                        "ink painting",
                        "cartoon",
                        "photograph"]
        self.model = difmodel
        self.df = pd.read_csv(dir)
        self.datadir = "/opt/data/private/stable_diffusion/my_origin_data_inter"
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        s = self.df.iloc[index]
        inter = inter_str_convert(s["inters"])
        prompts = s["prompt"]
        prompt_head = "A realism and extremely detailed "
        prompt_mid = " of a complete "
        c_list = list()
        for style in self.styles:
            c1 = self.model.get_learned_conditioning(4*[prompt_head + style + prompt_mid + prompts.split("&")[0]])
            c2 = self.model.get_learned_conditioning(4*[prompt_head + style + prompt_mid + prompts.split("&")[1]])
            c = torch.cat((c1, c2), dim=0).to(self.device)
            c_list.append(c)

        # c = torch.load(os.path.join(self.datadir, prompts, "weight", prompts + ".t")).to(self.device)
        # print(c.type())
        # print(inter.type())
        # inter = torch.load(os.path.join(self.datadir, prompts, "weight", prompts + "_best_inter.t")).to(
        #     torch.float32).to(self.device)
        # print(inter)
        return prompts, c_list, inter[0]


class my_clip_classifi_set(Dataset):
    def __init__(self, train=True, device="cuda:0"):
        if train:
            self.prompts, self.results = load_perm(f"df_results/interset/train.csv")
        else:
            self.prompts, self.results = load_perm(f"df_results/interset/test.csv")
        self.device = device

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompts = self.prompts[index].split(" and ")
        pos = self.poses[index]

        c1 = torch.load("weights_my/" + prompts[0] + ".t").to(self.device)[0].unsqueeze(0)
        c2 = torch.load("weights_my/" + prompts[1] + ".t").to(self.device)[0].unsqueeze(0)
        c = torch.cat((c1, c2), dim=0)
        inter = genInter_fromList(pos)
        # print(inter)
        return c, torch.as_tensor(inter, dtype=torch.float32, device=self.device)


def train_percol():
    LR = 0.2
    EPOCH = 1000

    # c1 = torch.load("weights_my/beaver on the plain.t").to(device)[0].unsqueeze(0)
    # c2 = torch.load("weights_my/turtle on the arctic.t").to(device)[0].unsqueeze(0)
    # c = torch.cat((c1, c2), dim=1)
    # interT = genRandomT_interpolate(1024, 0.5).to(device)
    dir = "weights_my"
    prompts = os.listdir(dir)
    snet = BSNet().to(device)
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snet.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(snet.parameters(), lr=LR, momentum=0.7, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 训练模型
    for epoch in range(EPOCH):
        c1pos = random.randint(0, len(prompts))
        c2pos = random.randint(0, len(prompts))
        interT = genRandomT_interpolate(1024, 0.5).to(device)

        c1 = torch.load("weights_my/" + prompts[c1pos]).to(device)
        c2 = torch.load("weights_my/" + prompts[c2pos]).to(device)
        c = torch.cat((c1, c2), dim=1)

        snet.train()

        loss_list = list()
        for iter in range(c.shape[2]):
            optimizer.zero_grad()
            this_c = c[:, :, iter].to(device)
            this_inter = interT[:, iter].unsqueeze(0).to(device)
            this_inter = torch.zeros((1, 2))
            # print(this_inter)
            this_inter[0, int(interT[:, iter])] = 1
            this_inter = this_inter.to(device)
            output_inter = snet(this_c)

            loss = criterion(output_inter, this_inter)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            # print(output_inter)
            # print("||---- iter: {0} ----|---- loss: {1} ----||".format(iter, round(loss.item(), 2)))
            if epoch == 10:
                print(output_inter, this_inter)
        scheduler.step()
        print("||---- epoch: {0} ----|---- loss: {1} ----||".format(epoch, round(mean(loss_list), 2)))


def train_single():
    dir = "weights_my"
    prompts = os.listdir(dir)
    c1pos = random.randint(0, len(prompts) - 1)
    c2pos = random.randint(0, len(prompts) - 1)
    interT = genRandomT_interpolate(1024, 0.5).to(device)
    c1 = torch.load("weights_my/" + prompts[c1pos]).to(device)[0].unsqueeze(0)
    c2 = torch.load("weights_my/" + prompts[c2pos]).to(device)[0].unsqueeze(0)
    c = torch.cat((c1, c2), dim=0)

    LR = 1e-2
    EPOCH = 1000

    snet = SNet().to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(snet.parameters(), lr=LR)
    optimizer = torch.optim.RMSprop(snet.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(snet.parameters(), lr=LR, momentum=0.7, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    snet.train()
    for epoch in range(EPOCH):
        # print(epoch)
        c = c.to(device)
        interT = interT.to(device)
        # print(c.shape)
        output_inter = snet(c)

        loss = criterion(output_inter, interT)
        loss.backward()
        optimizer.step()
        # print(round(loss.item(), 3))
        sum = 0
        truesum = 0

        for i in range(output_inter.shape[1]):
            # print(output_inter[0, i])
            # print(interT[0, i])
            if (output_inter[0, i] == 1 and interT[0, i] == 1) or (output_inter[0, i] != 1 and interT[0, i] != 1):
                sum += 1
            if (output_inter[0, i] == interT[0, i]):
                truesum += 1
            # print(i, output_inter[0, i], interT[0, i])
        truesum /= output_inter.shape[1]
        sum /= output_inter.shape[1]
        print("||-- loss: {0} --|-- acc: {1} --||-- tacc: {2} --||".format(round(loss.item(), 3), round(sum, 3),
                                                                           round(truesum, 3)))
        if epoch == 100:
            print(output_inter)
            print(interT)
    scheduler.step()
    # if epoch == 10:
    #     print(output_inter)
    #     print(interT)


def train_dis(device, isTrain):
    # python getPerms.py --cuda = 1 --fix = 0 --total = 3 --part = 2 --path = genPerms

    # savedir = './log/'
    # os.makedirs(savedir, exist_ok=True)
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(log_dir=savedir)
    BATCH_SIZE = 16
    LR = 1e-3
    EPOCH = 500
    mumodel = SNet().to(device)
    thetamodel = SNet().to(device)

    # criterion = CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer1 = torch.optim.RMSprop(mumodel.parameters(), lr=LR)
    optimizer2 = torch.optim.RMSprop(thetamodel.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.7, weight_decay=1e-5)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.99)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.99)
    trainset = my_inter_set_dis(train=isTrain)
    # testset = myPermset('df_results/'+dir, train=False)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    loss_list1 = list()
    loss_list2 = list()

    for current_epoch in range(EPOCH):
        mumodel.train()
        thetamodel.train()
        losses1 = list()
        losses2 = list()
        for idx, (prompt, train_c, train_mu, train_theta) in enumerate(train_loader):
            # print(idx)
            # print(train_c.shape, train_mu.shape, train_theta.shape)
            train_c, train_mu, train_theta = train_c.to(device), train_mu.to(device), train_theta.to(device)
            print(train_mu)
            # train_c.requires_grad=True
            # train_perm.requires_grad=True

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # print(train_c.grad_fn)
            this_mu = mumodel(train_c)
            this_theta = thetamodel(train_c)
            # print(this_theta.shape)
            # print(train_theta.shape)

            loss1 = criterion(this_mu, train_mu)
            loss2 = criterion(this_theta, train_theta)
            # loss = criterion(train_perm, perm).requires_grad_(True)

            loss1.backward()
            loss2.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            optimizer1.step()
            optimizer2.step()
            # scheduler.step()
            # sum = 0
            # truesum = 0
            # length = this_inter.shape[1]
            # for i in range(length):
            #     # print(output_inter[0, i])
            #     # print(interT[0, i])
            #     if (this_inter[0, i] == 1 and train_inter[0, i] == 1) or\
            #             (this_inter[0, i] != 1 and train_inter[0, i] != 1):
            #         sum += 1
            #     if (this_inter[0, i] == train_inter[0, i]):
            #         truesum += 1
            # truesum /= length
            # sum /= length
            # if current_epoch == 999:
            #     f = open("df_results/interset/inter.txt", mode="w+")
            #
            #     print(this_inter, train_inter)
            #     print(this_inter, train_inter, file=f)
            #     f.close()

            # print("||-- acc: {0} --||-- tacc: {1} --||".format(round(sum, 3), round(truesum, 3)))
        scheduler1.step()
        print("||-- epoch: {0} --|-- mu_loss: {1} --|-- theta_loss: {2} --||".
              format(current_epoch, round(mean(losses1), 3), round(mean(losses2), 3)))
        loss_list1.append(mean(losses1))
        loss_list2.append(mean(losses1))
    torch.save(mumodel.state_dict(), 'models/models/inter_mu.pkl')
    torch.save(mumodel.state_dict(), 'models/models/inter_theta.pkl')
    draw(loss_list1, 'models/loss_fig/inter_mu.jpg')
    draw(loss_list2, 'models/loss_fig/inter_theta.jpg')

    print("save finish!")


def train_interp(datadir, pkldir):
    seed = 25
    accelerator = Accelerator()
    set_seed(seed)
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCH = 1000
    model = SNet_deep()
    # criterion = CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.7, weight_decay=1e-5)

    # optimizer = nn.DataParallel(optimizer, device_ids=[0, 1, 2, 3])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    trainset = my_inter_set(datadir)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # train_loader = torch.utils.data.DataLoader(
    #     trainset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     sampler=train_sampler)
    #
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
    model, optimizer, train_data = accelerator.prepare(model, optimizer, train_loader)

    loss_list = list()

    for current_epoch in trange(EPOCH):
        model.train()
        losses = list()
        for idx, (_, train_c, train_inter) in enumerate(train_data):
            # train_c, train_inter = train_c.to(device), train_inter.to(device)

            optimizer.zero_grad()
            this_inter = model(train_c)
            # print(train_inter)
            # print(this_inter)

            loss = criterion(this_inter, train_inter)
            # loss = criterion(train_perm, perm).requires_grad_(True)

            # loss.backward()
            accelerator.backward(loss)

            losses.append(loss.item())
            optimizer.step()
            # scheduler.step()
        # accelerator.print(f"epoch【{current_epoch}】@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")
        if accelerator.is_main_process:
            accelerator.print("||-- epoch: {0} --|-- loss: {1} --||".
                  format(current_epoch, round(mean(losses), 3)))
        loss_list.append(mean(losses))
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        accelerator.save(unwrapped_model.state_dict(), pkldir)
    draw(loss_list, 'models/loss_fig/inter_deep.jpg')

    print("save finish!")


def test(pkldir, dir):
    from getPerms_interpolate import init
    seed = 25
    accelerator = Accelerator()
    set_seed(seed)
    device = accelerator.device
    difmodel, sampler, clipmodel, preprocess = init(device)
    BATCH_SIZE = 1
    model = SNet_deep().to(device)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(pkldir))

    # model.load_state_dict(torch.load(pkldir))
    model.eval()
    # testset = my_inter_set_style(dir, device, difmodel)
    testset = my_inter_set(dir, device)
    train_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    for idx, (prompts, train_c, train_inter) in enumerate(train_loader):
        prompts = str(prompts[0])

        # if get_single_hypersynsets(prompts.split("&")[0].replace(" ", "_")) in animal_class or get_single_hypersynsets(prompts.split("&")[1].replace(" ", "_")) in animal_class:
        #     continue
        print(prompts.split("&"))
        train_c, train_inter = train_c, train_inter
        this_inter = model(train_c)
        # median = this_inter.median().item()
        # this_inter[this_inter < median] = 0.0
        # this_inter[this_inter >= median] = 1.0
        samples_m = torch.zeros([2, 1024])
        samples_m[0] = train_inter
        samples_m[1] = this_inter

        test_inter(50, "./my_test_inters_show_0509", prompts, difmodel,
                   sampler, clipmodel, preprocess, device=device,
                   inters=samples_m, train_c=train_c.squeeze(0))


def test_ood(pkldir, dir):
    from getPerms_interpolate import init
    # accelerator = Accelerator()
    df = pd.read_csv("./df_results/inters/all_best.csv")
    demodf = pd.read_csv("./df_results/inters/all_best_demo.csv")
    demo_pr = demodf["prompt"]
    ood_df = pd.DataFrame(columns=demodf.columns)
    # device=accelerator.device
    device = "cuda:1"
    t = 0
    for i in range(300):
        while df.iloc[t]["prompt"] in demo_pr:
            t += 1
        ood_df = ood_df.append(df.iloc[i], ignore_index=True)
        t += 1
    ood_df.to_csv(dir, index=False)
    difmodel, sampler, clipmodel, preprocess = init(device)
    BATCH_SIZE = 1
    model = SNet_deep().to(device)

    # unwrap_model = accelerator.unwrap_model(model)
    # unwrap_model = accelerator.prepare(unwrap_model)
    # unwrap_model.load_state_dict(torch.load(pkldir))

    model.load_state_dict(torch.load(pkldir))
    model.eval()
    # unwrap_model.eval()

    testset = my_inter_set(dir)
    train_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    for idx, (prompts, train_c, train_inter) in enumerate(train_loader):
        prompts = str(prompts[0])
        print(prompts.split("&"))
        train_c, train_inter = train_c.to(device), train_inter.to(device)
        # this_inter = unwrap_model(train_c)
        this_inter = model(train_c)
        # median = this_inter.median().item()
        # this_inter[this_inter < median] = 0.0
        # this_inter[this_inter >= median] = 1.0
        samples_m = torch.zeros([2, 1024])
        samples_m[0] = train_inter
        samples_m[1] = this_inter

        test_inter(50, "./my_test_inters_demo", prompts, difmodel,
                   sampler, clipmodel, preprocess, device=device,
                   inters=samples_m, train_c=train_c.squeeze(0))

def test_style(pkldir, dir):
    from getPerms_interpolate import init
    seed = 25
    accelerator = Accelerator()
    set_seed(seed)
    device = "cuda:1"
    difmodel, sampler, clipmodel, preprocess = init(device)
    BATCH_SIZE = 1
    model = SNet_deep().to(device)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(pkldir))

    # model.load_state_dict(torch.load(pkldir))
    model.eval()
    # testset = my_inter_set_style(dir, device, difmodel)
    testset = my_inter_set_style(dir, device, difmodel)
    train_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    for idx, (prompts, train_c_list, train_inter) in enumerate(train_loader):
        prompts = str(prompts[0])
        # if prompts not in check_prompt:
        #     continue
        print(prompts.split("&"))
        # train_c, train_inter = train_c, train_inter
        samples_m_list = list()
        for train_c in train_c_list:
            this_inter = model(train_c)
            samples_m = torch.zeros([2, 1024])
            samples_m[0] = train_inter
            samples_m[1] = this_inter
            samples_m_list.append(samples_m)

        test_inter_style(50, "./my_test_inters_style", prompts, difmodel,
                   sampler, clipmodel, preprocess, device=device,
                   train_inter_list=samples_m_list, train_c_list=train_c_list)
def test_final(pkldir, dir):
    from getPerms_interpolate import init
    seed = 25
    accelerator = Accelerator()
    set_seed(seed)
    device = accelerator.device
    difmodel, sampler, clipmodel, preprocess = init(device)
    BATCH_SIZE = 1
    model = SNet_deep().to(device)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(pkldir))

    # model.load_state_dict(torch.load(pkldir))
    model.eval()
    # testset = my_inter_set(dir, device, difmodel)
    testset = my_inter_set_style(dir, device, difmodel)
    train_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    for idx, (prompts, train_c_list, train_inter) in enumerate(train_loader):
        prompts = str(prompts[0])
        if prompts not in best_prompt:
            continue
        print(prompts.split("&"))
        # train_c, train_inter = train_c, train_inter
        samples_m_list = list()
        for train_c in train_c_list:
            this_inter = model(train_c)
            samples_m = torch.zeros([2, 1024])
            samples_m[0] = train_inter
            samples_m[1] = this_inter
            samples_m_list.append(samples_m)

        test_inter_style(50, "./my_test_inters_show_check", prompts, difmodel,
                   sampler, clipmodel, preprocess, device=device,
                   train_inter_list=samples_m_list, train_c_list=train_c_list)

def test_inter(s=50, out_path="genTs", prompt_tails=None, model=None,
               sampler=None, clipmodel=None, preprocess=None, device="cpu", inters=None, train_c=None):
    from getPerms_interpolate import img_prompt_simi
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    batch_size = 6
    os.makedirs(out_path, exist_ok=True)

    prompt_head = "A realistic and extremely detailed photograph of a complete "
    prompt_head = "A realistic and extremely detailed photograph of a complete "
    prompt_half_head = "A realistic and extremely detailed photograph of a new thing of the hybrid of "
    prompts_l = prompt_tails.split("&")
    prompt_ablation = prompt_half_head + prompts_l[0] + " and " + prompts_l[1]
    prompt = prompt_head + prompts_l[0] + " and " + prompts_l[1]
    c_ablation = model.get_learned_conditioning(batch_size * [prompt_ablation]).to(device)

    c0 = train_c[:4].to(device)
    c1 = train_c[4:].to(device)
    this_inter = inters[1].to(device)
    this_inter = torch.clamp(this_inter, 0).to(device)
    train_inter = inters[0].to(device)
    c = torch.zeros([batch_size, 77, 1024]).to(device)
    this_rev_inter = (torch.ones_like(this_inter) - this_inter).to(device)
    train_rev_inter = (torch.ones_like(train_inter) - train_inter).to(device)

    c[0] = torch.mul(c0[0], this_inter) + torch.mul(c1[0], this_rev_inter)
    c[5] = torch.mul(c0[0], this_rev_inter) + torch.mul(c1[0], this_inter)
    c[1] = torch.mul(c0[0], train_inter) + torch.mul(c1[0], train_rev_inter)
    c[2] = c0[0]
    c[3] = c1[0]
    c[4] = c_ablation[0]
    c = c.to(device)

    precision_scope = autocast
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        uc = model.get_learned_conditioning(batch_size * [""]).to(device)
        shape = [supC, supH // supf, supW // supf]
        samples, _ = sampler.sample(S=s,
                                    conditioning=c,
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=9.0,
                                    unconditional_conditioning=uc,
                                    eta=0.0,
                                    x_T=None)

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        imgs = list()
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            imgs.append(img)
        text_input_0 = clip.tokenize(prompt_head + prompts_l[0]).to(device)
        text_input_1 = clip.tokenize(prompt_head + prompts_l[1]).to(device)
        simi_m_0 = img_prompt_simi(imgs[0], text_input_0, clipmodel, preprocess, device)
        simi_m_1 = img_prompt_simi(imgs[0], text_input_1, clipmodel, preprocess, device)
        simi_t_0 = img_prompt_simi(imgs[1], text_input_0, clipmodel, preprocess, device)
        simi_t_1 = img_prompt_simi(imgs[1], text_input_1, clipmodel, preprocess, device)
        # if abs(simi_m_0 - simi_m_1) <= 0.03:
        imgs[0].save(os.path.join(out_path, f"{prompt_tails}_model_{(round(simi_m_0, 2))}_{round(simi_m_1, 2)}.png"))
        imgs[5].save(os.path.join(out_path, f"{prompt_tails}_model_inv.png"))
        imgs[1].save(os.path.join(out_path, f"{prompt_tails}_train_{(round(simi_t_0, 2))}_{round(simi_t_1, 2)}.png"))
        imgs[2].save(os.path.join(out_path, f"{prompt_tails}_{prompts_l[0]}.png"))
        imgs[3].save(os.path.join(out_path, f"{prompt_tails}_{prompts_l[1]}.png"))
        imgs[4].save(os.path.join(out_path, f"{prompt_tails}_baseline.png"))


def test_inter_style(s=50, out_path="genTs", prompt_tails=None, model=None,
               sampler=None, clipmodel=None, preprocess=None, device="cpu", train_inter_list=None, train_c_list=None):
    from getPerms_interpolate import img_prompt_simi
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    batch_size = 5
    styles = ["Aquarell",
              "line art",
              "ink painting",
              "cartoon",
              "photograph"]
    os.makedirs(out_path, exist_ok=True)
    c = torch.zeros([batch_size, 77, 1024]).to(device)
    for i_style in range(len(train_c_list)):
        train_c = train_c_list[i_style].squeeze(0)
        inters = train_inter_list[i_style]
        prompts_l = prompt_tails.split("&")

        c0 = train_c[:4].to(device)
        c1 = train_c[4:].to(device)
        this_inter = inters[1].to(device)
        this_inter = torch.clamp(this_inter, 0).to(device)
        train_inter = inters[0].to(device)

        this_rev_inter = (torch.ones_like(this_inter) - this_inter).to(device)
        train_rev_inter = (torch.ones_like(train_inter) - train_inter).to(device)

        c[i_style] = torch.mul(c0[0], this_inter) + torch.mul(c1[0], this_rev_inter)
    c = c.to(device)

    precision_scope = autocast

    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        uc = model.get_learned_conditioning(batch_size * [""]).to(device)
        shape = [supC, supH // supf, supW // supf]
        samples, _ = sampler.sample(S=s,
                                    conditioning=c,
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=9.0,
                                    unconditional_conditioning=uc,
                                    eta=0.0,
                                    x_T=None)

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        imgs = list()
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            imgs.append(img)
        prompt_head = "A realism and extremely detailed "
        prompt_mid = " of a complete "

        text_input_0 = clip.tokenize(prompt_head + styles[4] + prompt_mid + prompts_l[0]).to(device)
        text_input_1 = clip.tokenize(prompt_head + styles[4] + prompt_mid + prompts_l[1]).to(device)
        simi_m_0 = img_prompt_simi(imgs[4], text_input_0, clipmodel, preprocess, device)
        simi_m_1 = img_prompt_simi(imgs[4], text_input_1, clipmodel, preprocess, device)
        if abs(simi_m_0 - simi_m_1) <= 0.05:
            for i_ofc in range(batch_size):
                imgs[i_ofc].save(os.path.join(out_path, f"{prompt_tails}_{styles[i_ofc]}.png"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():


    train_interp("./df_results/inters/all_best.csv", 'models/models/inter_deep.pkl', "multi")


if __name__ == '__main__':
    # device = torch.device("cuda:1")
    # make_dataset(0.1)
    # train_interp("./df_results/inters/all_best.csv", 'models/models/inter_deep.pkl')
    test('models/models/inter_deep.pkl',"./df_results/inters/all_best.csv")
    # test_style('models/models/inter_deep.pkl',"./df_results/inters/all_best.csv")
    # test_final('models/models/inter_deep.pkl',"./df_results/inters/all_best.csv")
    # test_ood('models/models/inter_deep_demo.pkl', "./df_results/inters/all_best_ood.csv")