import os
import random

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch import autocast, nn
from torch.utils.data import DataLoader, Dataset

import clip
from PIL import Image
from einops import rearrange

from train_interpolate import inter_str_convert


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test(prompt1, prompt2):
    from getPerms_interpolate import init
    os.makedirs("./my_output", exist_ok=True)
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    seed = 25
    s = 50
    batch_size = 5
    accelerator = Accelerator()
    set_seed(seed)
    device = accelerator.device
    difmodel, sampler, clipmodel, preprocess = init(device)
    BATCH_SIZE = 1
    model = SNet_deep().to(device)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load('models/models/inter_deep.pkl'))

    # model.load_state_dict(torch.load(pkldir))
    model.eval()

    prompt_head = "A photograph of a complete "
    prompt_base = "A photograph of hybrid of "
    c1 = difmodel.get_learned_conditioning(4 * [prompt_head + prompt1]).to(device)
    c2 = difmodel.get_learned_conditioning(4 * [prompt_head + prompt2]).to(device)
    c3 = difmodel.get_learned_conditioning(4 * [prompt_base + prompt1 + " and " + prompt2]).to(device)
    cc = torch.cat((c1, c2), dim=0).to(device).unsqueeze(0)
    this_inter = model(cc).to(device)
    this_rev_inter = (torch.ones_like(this_inter) - this_inter).to(device)

    c_mul = torch.mul(c1, this_inter) + torch.mul(c2, this_rev_inter).to(device)
    c_mul_rev = torch.mul(c2, this_inter) + torch.mul(c1, this_rev_inter).to(device)

    c = torch.zeros([5, 77, 1024]).to(device)
    c[0] = c_mul[0]
    c[1] = c_mul_rev[0]
    c[2] = c1[0]
    c[3] = c2[0]
    c[4] = c3[0]
    c = c.to(device)
    precision_scope = autocast
    with torch.no_grad(), precision_scope("cuda"), difmodel.ema_scope():
        uc = difmodel.get_learned_conditioning(batch_size * [""]).to(device)
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

        x_samples = difmodel.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        i = 0
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            if i == 0:
                img.save(os.path.join("./my_output", prompt1 + "&" + prompt2 + "_model.png"))
            if i == 1:
                img.save(os.path.join("./my_output", prompt1 + "&" + prompt2 + "_model_rev.png"))
            if i == 2:
                img.save(os.path.join("./my_output", prompt1 + "&" + prompt2 + "_p1.png"))
            if i == 3:
                img.save(os.path.join("./my_output", prompt1 + "&" + prompt2 + "_p2.png"))
            if i == 4:
                img.save(os.path.join("./my_output", prompt1 + "&" + prompt2 + "_baseline.png"))
            i += 1
    return this_inter.detach().cpu()

def test_inter(prompt1, prompt2):
    from getPerms_interpolate import init
    os.makedirs("my_output_inter", exist_ok=True)
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    seed = 25
    s = 50
    batch_size = 5
    accelerator = Accelerator()
    set_seed(seed)
    device = accelerator.device
    difmodel, sampler, clipmodel, preprocess = init(device)
    BATCH_SIZE = 1
    model = SNet_deep().to(device)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load('models/models/inter_deep.pkl'))

    # model.load_state_dict(torch.load(pkldir))
    model.eval()

    prompt_head = "A photograph of a complete "
    prompt_base = "A photograph of hybrid of "
    c1 = difmodel.get_learned_conditioning(4 * [prompt_head + prompt1]).to(device)
    c2 = difmodel.get_learned_conditioning(4 * [prompt_head + prompt2]).to(device)
    c3 = difmodel.get_learned_conditioning(4 * [prompt_base + prompt1 + " and " + prompt2]).to(device)
    cc = torch.cat((c1, c2), dim=0).to(device).unsqueeze(0)
    this_inter = model(cc).to(device)
    this_rev_inter = (torch.ones_like(this_inter) - this_inter).to(device)
    c_mul = torch.mul(c1, this_inter) + torch.mul(c2, this_rev_inter).to(device)
    c_mul_rev = torch.mul(c2, this_inter) + torch.mul(c1, this_rev_inter).to(device)

    c = torch.zeros([5, 77, 1024]).to(device)
    for i in range(5):
        inter = random.uniform(0, 1)
        c[i] = c1[0] * inter + c2[0] * (1 - inter)
    # c[0] = c_mul[0]
    # c[1] = c_mul_rev[0]
    # c[2] = c1[0]
    # c[3] = c2[0]
    # c[4] = c3[0]
    c = c.to(device)
    precision_scope = autocast
    with torch.no_grad(), precision_scope("cuda"), difmodel.ema_scope():
        uc = difmodel.get_learned_conditioning(batch_size * [""]).to(device)
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

        x_samples = difmodel.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        i = 0
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))

            img.save(os.path.join("./my_output_inter", prompt1 + "&" + prompt2 + "_" + str(i) + ".png"))
            i += 1


def attribute_test():
    # prompts = os.listdir("/opt/data/private/stable_diffusion/my_eval_check")
    prompts = ["CD player&beer glass",
               "mushroom&poppies",
               "feather boa&ostrich",
               "macaque&timber wolf",
               "zucchini&vulture",
               "jackfruit&thresher",
               "Australian terrier&tiger",
               "toucan&bathing cap",
               "American alligator&Arabian camel",
               "pineapple&bagel",
               "king crab&orangutan",
               "lobster&sea slug",
               "lionfish&abacus",
               "kangaroo&pears",
               "subflower&orange"]
    advs = ["blue", "green", "flying", "jumping"]
    for i in range(50):
        prompt_l = prompts[i].split("&")
        for adv1 in advs:
            for adv2 in advs:
                test(adv1 + " " + prompt_l[0], adv2 + " " + prompt_l[1])


def Inter_test():
    prompts = ["sunflower&orange",
               "kangaroo&pears"]
    for i in prompts:
        prompt_l = i.split("&")
        test_inter(prompt_l[0], prompt_l[1])


if __name__ == '__main__':
    # test("African crocodile", "sundial")
    Inter_test()