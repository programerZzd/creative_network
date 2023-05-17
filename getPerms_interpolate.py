import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import random
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image
from wordnet import get_single_hypersynsets

from clip import clip
from einops import rearrange
from omegaconf import OmegaConf
from torch import autocast
from tqdm import trange

from clipTest import clipClassification
from getPerms import partial_init
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from scripts.txt2img import load_model_from_config

torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
class_animal = ('camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk')
class_plant = ('orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
               'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers')
class_humanmade = ('clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                   'bed', 'chair', 'couch', 'table', 'wardrobe')


def init(device):
    # if opt.cuda:
    #     device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    # else:
    #     device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    super_config = "configs/stable-diffusion/v2-inference.yaml"
    super_ckpt = "ldm/512-base-ema.ckpt"
    config = OmegaConf.load(f"{super_config}")

    model = load_model_from_config(config, f"{super_ckpt}")
    model = model.to(device)
    model.cond_stage_model.device = device
    sampler = DPMSolverSampler(model)
    for m in model.parameters():
        m.requires_grad = False
    clipmodel, preprocess = clip.load('ViT-L/14@336px', device)
    for m in clipmodel.parameters():
        m.requires_grad = False
    return model, sampler, clipmodel.to(device), preprocess


def img_prompt_simi(img, text_input, clipmodel, pre, device):
    with torch.no_grad():
        # text_input = clip.tokenize(prompt).to(device)

        image_features = clipmodel.encode_image(pre(img).unsqueeze(0).to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = clipmodel.encode_text(text_input)
        norm_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # simi = torch.cosine_similarity(image_features, text_features)
        norm_simi = torch.cosine_similarity(image_features, norm_features)
        return norm_simi.item()


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
    return torch.tensor(T).type(torch.half)


# 多句生成
# 对比实验
# 画图

def getT_interpolate(s=50, out_path="genTs", prompt_tails=None, model=None,
                     sampler=None, clipmodel=None, preprocess=None, uc=None,
                     x_T=None, device="cpu"):
    prompt_head = "A realistic and extremely detailed photograph of a complete "
    prompt_half_head = "A realistic and extremely detailed photograph of "
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    batch_size = 4
    os.makedirs(out_path, exist_ok=True)

    out_path = os.path.join(out_path, prompt_tails)
    os.makedirs(out_path, exist_ok=True)

    os.makedirs(out_path, exist_ok=True)

    out_normal_path = os.path.join(out_path, "normal/")
    out_inter_path = os.path.join(out_path, "inter/")
    out_weight_path = os.path.join(out_path, "weight/")
    os.makedirs(out_normal_path, exist_ok=True)
    os.makedirs(out_inter_path, exist_ok=True)
    os.makedirs(out_weight_path, exist_ok=True)
    print("finish!")

    base_count = len(os.listdir(out_normal_path))

    precision_scope = autocast
    prompt_tail = prompt_tails.split("&")
    zero_Ts = list()
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        if uc == None:
            uc = model.get_learned_conditioning(batch_size * [""]).to(device)
        c1 = model.get_learned_conditioning(batch_size * [prompt_head + prompt_tail[0]]).to(device)
        c2 = model.get_learned_conditioning(batch_size * [prompt_head + prompt_tail[1]]).to(device)
        c3 = model.get_learned_conditioning(
            batch_size * [prompt_head + prompt_tail[0] + " and " + prompt_head + prompt_tail[1]]).to(device)
        shape = [supC, supH // supf, supW // supf]
        file_list = list()
        inter_list = list()

        shuffle = 0.5
        for rrr in trange(100, desc=prompt_tails):
            c = torch.zeros_like(c1).to(device)

            inter_Ts = list()
            for _ in range(batch_size):
                interT = genRandomT_interpolate(c.shape[2], shuffle).to(device)
                rev_interT = (torch.ones_like(interT) - interT).to(device)
                c[_] = torch.mul(c1[_], interT) + torch.mul(c2[_], rev_interT)
                inter_Ts.append(interT)
            # c[2] = c3[2]
            # c[3] = c3[3]

            samples, _ = sampler.sample(S=s,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=9.0,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=x_T)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            img_list = list()

            for x_sample, i in zip(x_samples, range(batch_size)):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                text_input_0 = clip.tokenize(prompt_head + prompt_tail[0]).to(device)
                text_input_1 = clip.tokenize(prompt_head + prompt_tail[1]).to(device)
                simi_0 = img_prompt_simi(img, text_input_0, clipmodel, preprocess, device)
                simi_1 = img_prompt_simi(img, text_input_1, clipmodel, preprocess, device)
                if abs(simi_0 - simi_1) <= 0.05:
                    img.save(
                        os.path.join(out_inter_path, f"{base_count:05}_{(round(simi_0, 2))}_{round(simi_1, 2)}.png"))
                    base_count += 1
                    inter_list.append(inter_Ts[i].detach().cpu().numpy())
                    # img_list.append(img)
                    file_list.append(f"{base_count:05}")

                # if i > 1:
                #     simi_0 = img_prompt_simi(img, prompt_head + prompt_tail[0], clipmodel, preprocess, device)
                #     simi_1 = img_prompt_simi(img, prompt_head + prompt_tail[1], clipmodel, preprocess, device)
                #
                #     img.save(
                #         os.path.join(out_normal_path, f"{base_count:05}_{(round(simi_0, 2))}_{round(simi_1, 2)}.png"))
                # else:
                #     simi_0 = img_prompt_simi(img, prompt_head + prompt_tail[0], clipmodel, preprocess, device)
                #     simi_1 = img_prompt_simi(img, prompt_head + prompt_tail[1], clipmodel, preprocess, device)
                #     img.save(os.path.join(out_inter_path, f"{base_count:05}_{(round(simi_0, 2))}_{round(simi_1, 2)}.png"))
                # base_count += 1

        df = pd.DataFrame(list(zip(file_list, inter_list)), columns=["img_name", "interpolate_position"])
        df.to_csv(os.path.join(out_weight_path, prompt_tails + ".csv"), index=False)
        with open(os.path.join(out_weight_path, prompt_tails + ".t"), 'wb') as f:
            torch.save(torch.cat((c1, c2), dim=0), f)


def getT_interpolate_single_check(s=50, out_path="genTs", prompt_tail=None, model=None,
                                  sampler=None, clipmodel=None, preprocess=None, uc=None,
                                  x_T=None, device="cpu"):
    prompt_head = "A realistic and extremely detailed photograph of a complete "
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    batch_size = 4
    os.makedirs(out_path, exist_ok=True)

    out_path = os.path.join(out_path, prompt_tail)
    os.makedirs(out_path, exist_ok=True)

    os.makedirs(out_path, exist_ok=True)

    out_normal_path = os.path.join(out_path, "normal/")
    out_inter_path = os.path.join(out_path, "inter/")
    out_weight_path = os.path.join(out_path, "weight/")
    os.makedirs(out_normal_path, exist_ok=True)
    os.makedirs(out_inter_path, exist_ok=True)
    os.makedirs(out_weight_path, exist_ok=True)
    print("finish!")

    base_count = len(os.listdir(out_normal_path))

    precision_scope = autocast
    zero_Ts = list()
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        if uc == None:
            uc = model.get_learned_conditioning(batch_size * [""]).to(device)
        c1 = model.get_learned_conditioning(batch_size * [prompt_head + prompt_tail]).to(device)

        shape = [supC, supH // supf, supW // supf]
        file_list = list()
        inter_list = list()

        shuffle = 0.5
        for rrr in trange(100, desc=prompt_tail):
            for ratio in range(4):
                c = torch.zeros_like(c1).to(device)

                c2 = torch.roll(c1, ratio * int(c.shape[1] / 4), 2)
                inter_Ts = list()

                for _ in range(batch_size):
                    interT = genRandomT_interpolate(c.shape[2], shuffle).to(device)
                    rev_interT = (torch.ones_like(interT) - interT).to(device)
                    c[_] = torch.mul(c1[_], interT) + torch.mul(c2[_], rev_interT)
                    inter_Ts.append(interT)
                # c[2] = c3[2]
                # c[3] = c3[3]

                samples, _ = sampler.sample(S=s,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=9.0,
                                            unconditional_conditioning=uc,
                                            eta=0.0,
                                            x_T=x_T)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                img_list = list()

                for x_sample, i in zip(x_samples, range(batch_size)):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    text_input = clip.tokenize(prompt_head + prompt_tail).to(device)
                    simi_0 = img_prompt_simi(img, text_input, clipmodel, preprocess, device)
                    _, clip_degree, chaos_degree = clipClassification(img, prompt_tail, clipmodel, preprocess,
                                                                      device)
                    if chaos_degree < 0.2:
                        img.save(os.path.join(out_path, f"{base_count:05}_{ratio}_{(round(simi_0, 2))}.png"))
                        inter_list.append(inter_Ts[i].detach().cpu().numpy())
                        file_list.append(f"{base_count:05}")
                        base_count += 1
            df = pd.DataFrame(list(zip(file_list, inter_list)), columns=["img_name", "interpolate_position"])
            df.to_csv(os.path.join(out_weight_path, prompt_tail + "_" + str(ratio) + ".csv"), index=False)
            with open(os.path.join(out_weight_path, prompt_tail + "_" + str(ratio) + ".t"), 'wb') as f:
                torch.save(torch.cat((c1, c2), dim=0), f)


def getT_interpolate_listPrompt(s=50, out_path="genTs", prompt_tails=None, model=None,
                                sampler=None, clipmodel=None, preprocess=None, uc=None,
                                x_T=None, device="cpu"):
    prompt_head = "A realistic and extremely detailed photograph of a complete "
    prompt_mix_head = "A realistic and extremely detailed photograph of the mixture of "
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    batch_size = 4
    os.makedirs(out_path, exist_ok=True)

    out_path = os.path.join(out_path, prompt_tails[0])
    os.makedirs(out_path, exist_ok=True)

    os.makedirs(out_path, exist_ok=True)

    out_normal_path = os.path.join(out_path, "normal/")
    out_inter_path = os.path.join(out_path, "inter/")
    out_weight_path = os.path.join(out_path, "weight/")
    os.makedirs(out_normal_path, exist_ok=True)
    os.makedirs(out_inter_path, exist_ok=True)
    os.makedirs(out_weight_path, exist_ok=True)
    print("finish!")

    base_count = len(os.listdir(out_normal_path))

    precision_scope = autocast
    # prompt_tail = prompt_tails.split("&")
    zero_Ts = list()
    base_c = None
    base_prompt = " "
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        while len(prompt_tails) != 0:
            now_prompt = prompt_tails.pop(0)
            if uc == None:
                uc = model.get_learned_conditioning(batch_size * [""]).to(device)

            now_c = model.get_learned_conditioning(batch_size * [prompt_head + now_prompt]).to(device)
            if base_c == None:
                base_c = now_c
                base_prompt = now_prompt
                continue
            shape = [supC, supH // supf, supW // supf]
            file_list = list()
            inter_list = list()

            shuffle = 0.5
            maxsimi = 0
            for rrr in trange(100):
                c = torch.zeros_like(base_c).to(device)

                inter_Ts = list()
                for _ in range(batch_size):
                    interT = genRandomT_interpolate(c.shape[2], shuffle).to(device)
                    rev_interT = (torch.ones_like(interT) - interT).to(device)
                    c[_] = torch.mul(base_c[_], interT) + torch.mul(now_c[_], rev_interT)
                    inter_Ts.append(interT)
                # c[2] = c3[2]
                # c[3] = c3[3]

                samples, _ = sampler.sample(S=s,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=9.0,
                                            unconditional_conditioning=uc,
                                            eta=0.0,
                                            x_T=x_T)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                img_list = list()

                for x_sample, i in zip(x_samples, range(batch_size)):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    text_input_0 = clip.tokenize(prompt_mix_head + base_prompt).to(device)
                    text_input_1 = clip.tokenize(prompt_head + now_prompt).to(device)
                    simi_0 = img_prompt_simi(img, text_input_0, clipmodel, preprocess, device)
                    simi_1 = img_prompt_simi(img, text_input_1, clipmodel, preprocess, device)
                    if abs(simi_0 - simi_1) <= 0.05:
                        img.save(
                            os.path.join(out_inter_path,
                                         f"{base_count:05}_{(round(simi_0, 2))}_{round(simi_1, 2)}.png"))
                        base_count += 1
                        inter_list.append(inter_Ts[i].detach().cpu().numpy())
                        # img_list.append(img)
                        file_list.append(f"{base_count:05}")
                        if (simi_0 + simi_1) / 2 >= maxsimi:
                            maxsimi = (simi_0 + simi_1) / 2
                            bestc = c[i].unsqueeze(0)
                            true_c = torch.vstack([bestc, bestc, bestc, bestc])

            df = pd.DataFrame(list(zip(file_list, inter_list)), columns=["img_name", "interpolate_position"])
            df.to_csv(os.path.join(out_weight_path, base_prompt + ".csv"), index=False)
            with open(os.path.join(out_weight_path, base_prompt + " & " + now_prompt + ".t"), 'wb') as f:
                torch.save(torch.cat((base_c, now_c, true_c), dim=0), f)
            with open(os.path.join(out_weight_path, base_prompt + " & " + now_prompt + "_inter.t"), 'wb') as f:
                torch.save(inter_Ts[i], f)

            base_prompt = base_prompt + " and " + now_prompt
            base_c = true_c


def getT_interpolate_ablation(s=50, out_path="genTs", prompt_tails=None, model=None,
                              sampler=None, clipmodel=None, preprocess=None, uc=None,
                              x_T=None, device="cpu"):
    prompt_head = "A realistic and extremely detailed photograph of a complete "
    prompt_half_head = "A realistic and extremely detailed photograph of "
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    batch_size = 4
    os.makedirs(out_path, exist_ok=True)

    out_path = os.path.join(out_path, prompt_tails)
    os.makedirs(out_path, exist_ok=True)

    os.makedirs(out_path, exist_ok=True)

    out_normal_path = os.path.join(out_path, "single_prompt/")
    out_mix_path = os.path.join(out_path, "mix_prompt/")
    out_inter_path = os.path.join(out_path, "inter/")
    out_weight_path = os.path.join(out_path, "weight/")
    os.makedirs(out_normal_path, exist_ok=True)
    os.makedirs(out_inter_path, exist_ok=True)
    os.makedirs(out_weight_path, exist_ok=True)
    os.makedirs(out_mix_path, exist_ok=True)
    print("finish!")

    base_count = len(os.listdir(out_normal_path))

    precision_scope = autocast
    prompt_tail = prompt_tails.split("&")
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        if uc == None:
            uc = model.get_learned_conditioning(batch_size * [""]).to(device)
        c1 = model.get_learned_conditioning(batch_size * [prompt_head + prompt_tail[0]]).to(device)
        c2 = model.get_learned_conditioning(batch_size * [prompt_head + prompt_tail[1]]).to(device)
        c3 = model.get_learned_conditioning(
            batch_size * [prompt_half_head + prompt_tail[0] + " and " + prompt_head + prompt_tail[1]]).to(device)
        shape = [supC, supH // supf, supW // supf]
        file_list = list()
        inter_list = list()

        shuffle = 0.5
        for rrr in trange(100, desc=prompt_tails):
            c = torch.zeros_like(c1).to(device)

            interT = genRandomT_interpolate(c.shape[2], shuffle).to(device)
            rev_interT = (torch.ones_like(interT) - interT).to(device)
            c[0] = torch.mul(c1[0], interT) + torch.mul(c2[0], rev_interT)
            c[1] = c1[0]
            c[2] = c2[0]
            c[3] = c3[0]

            samples, _ = sampler.sample(S=s,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=9.0,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=x_T)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            img_list = list()

            for x_sample in x_samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img_list.append(img)
            text_input_0 = clip.tokenize(prompt_head + prompt_tail[0]).to(device)
            text_input_1 = clip.tokenize(prompt_head + prompt_tail[1]).to(device)
            simi_0 = img_prompt_simi(img_list[0], text_input_0, clipmodel, preprocess, device)
            simi_1 = img_prompt_simi(img_list[0], text_input_1, clipmodel, preprocess, device)
            if abs(simi_0 - simi_1) <= 0.05:
                img_list[0].save(
                    os.path.join(out_inter_path, f"{base_count:05}_{(round(simi_0, 2))}_{round(simi_1, 2)}.png"))
                img_list[1].save(
                    os.path.join(out_normal_path, f"{base_count:05}_0.png"))
                img_list[2].save(
                    os.path.join(out_normal_path, f"{base_count:05}_1.png"))
                img_list[3].save(
                    os.path.join(out_mix_path, f"{base_count:05}.png"))

                base_count += 1
                inter_list.append(interT.detach().cpu().numpy())
                # img_list.append(img)
                file_list.append(f"{base_count:05}")

        df = pd.DataFrame(list(zip(file_list, inter_list)), columns=["img_name", "interpolate_position"])
        df.to_csv(os.path.join(out_weight_path, prompt_tails + ".csv"), index=False)
        with open(os.path.join(out_weight_path, prompt_tails + ".t"), 'wb') as f:
            torch.save(torch.cat((c1, c2), dim=0), f)


def getT_interpolate_baseline(s=50, out_path="genTs", prompt_tails=None, model=None,
                              sampler=None, clipmodel=None, preprocess=None, uc=None,
                              x_T=None, device="cuda:0"):
    prompt_head = "A realistic photograph of a "
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    os.makedirs(out_path, exist_ok=True)

    out_path = os.path.join(out_path, prompt_tails)
    os.makedirs(out_path, exist_ok=True)

    os.makedirs(out_path, exist_ok=True)

    out_normal_path = os.path.join(out_path, "normal/")
    out_inter_path = os.path.join(out_path, "inter/")
    os.makedirs(out_normal_path, exist_ok=True)
    os.makedirs(out_inter_path, exist_ok=True)

    batch_size = 4
    base_count = len(os.listdir(out_normal_path))

    precision_scope = autocast
    prompt_tail = prompt_tails.split("&")
    zero_Ts = list()
    prompt_list = ['mixture', 'hybrid', 'cross', 'interpolate']
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        if uc == None:
            uc = model.get_learned_conditioning(batch_size * [""]).to(device)
        clist = list()
        for pl in range(len(prompt_list)):
            clist.append(model.get_learned_conditioning(batch_size * [
                prompt_head + prompt_list[pl] + " of " + prompt_tail[0] + " and " + prompt_head + prompt_tail[1]]).to(
                device))

        shape = [supC, supH // supf, supW // supf]
        shuffle = 0.5
        for _ in trange(5, desc=prompt_tails):
            c = torch.zeros([4, 77, 1024]).to(device)
            for _ in range(4):
                c[_] = clist[_][0]
            samples, _ = sampler.sample(S=s,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=9.0,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=x_T)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            img_list = list()

            for x_sample, i in zip(x_samples, range(4)):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                simi_0 = img_prompt_simi(img, prompt_head + prompt_tail[0], clipmodel, preprocess, device)
                simi_1 = img_prompt_simi(img, prompt_head + prompt_tail[1], clipmodel, preprocess, device)
                img.save(os.path.join(out_normal_path,
                                      f"{base_count:05}_{prompt_list[i]}_{(round(simi_0, 2))}_{round(simi_1, 2)}.png"))
                base_count += 1
                # img_list.append(img)


def getT_interpolate_classes(s=50, out_path="genTs", prompt_tails=None, model=None,
                             sampler=None, clipmodel=None, preprocess=None, uc=None,
                             x_T=None, device="cuda:0"):
    prompt_head = "A realistic and extremely detailed photograph of a complete "
    prompt_half_head = "A realistic and extremely detailed photograph of "
    supC = 4
    supH = 512
    supW = 512
    supf = 8
    batch_size = 4
    origin_out_path = os.path.join("/opt/data/private/stable_diffusion/", out_path)
    target_out_path = "/opt/data/private/stable_diffusion/my_origin_data_inter"
    os.makedirs(origin_out_path, exist_ok=True)
    origin_out_path = os.path.join(origin_out_path, prompt_tails)
    target_out_path = os.path.join(target_out_path, prompt_tails)

    os.makedirs(origin_out_path, exist_ok=True)


    origin_normal_path = os.path.join(origin_out_path, "normal/")
    target_normal_path = os.path.join(target_out_path, "normal/")
    origin_inter_path = os.path.join(origin_out_path, "inter/")
    target_inter_path = os.path.join(target_out_path, "inter/")
    out_weight_path = os.path.join(target_out_path, "weight/")
    os.makedirs(origin_normal_path, exist_ok=True)

    os.makedirs(origin_inter_path, exist_ok=True)

    print("finish!")

    base_count = len(os.listdir(origin_normal_path))

    precision_scope = autocast
    prompt_tail = prompt_tails.split("&")
    zero_Ts = list()
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        if uc == None:
            uc = model.get_learned_conditioning(batch_size * [""]).to(device)
        c1 = model.get_learned_conditioning(batch_size * [prompt_head + prompt_tail[0]]).to(device)
        c2 = model.get_learned_conditioning(batch_size * [prompt_head + prompt_tail[1]]).to(device)
        c3 = model.get_learned_conditioning(
            batch_size * [prompt_head + prompt_tail[0] + " and " + prompt_head + prompt_tail[1]]).to(device)
        shape = [supC, supH // supf, supW // supf]
        file_list = list()
        inter_list = list()

        shuffle = 0.5
        for rrr in trange(50, desc=prompt_tails):
            c = torch.zeros_like(c1).to(device)

            inter_Ts = list()
            for _ in range(batch_size):
                interT = genRandomT_interpolate(c.shape[2], shuffle).to(device)
                rev_interT = (torch.ones_like(interT) - interT).to(device)
                c[_] = torch.mul(c1[_], interT) + torch.mul(c2[_], rev_interT)
                inter_Ts.append(interT)
            # c[2] = c3[2]
            # c[3] = c3[3]

            samples, _ = sampler.sample(S=s,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=9.0,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=x_T)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            img_list = list()

            for x_sample, i in zip(x_samples, range(batch_size)):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                # img.save(os.path.join(origin_inter_path, f"{base_count:05}.png"))
                # inter_list.append(interT.detach().cpu().numpy())
                # file_list.append(f"{base_count:05}")

                text_input_0 = clip.tokenize(prompt_head + prompt_tail[0]).to(device)
                text_input_1 = clip.tokenize(prompt_head + prompt_tail[1]).to(device)
                simi_0 = img_prompt_simi(img, text_input_0, clipmodel, preprocess, device)
                simi_1 = img_prompt_simi(img, text_input_1, clipmodel, preprocess, device)
                if abs(simi_0 - simi_1) <= 0.05:
                    img.save(
                        os.path.join(origin_inter_path, f"{base_count:05}_{(round(simi_0, 2))}_{round(simi_1, 2)}.png"))
                    base_count += 1
                    inter_list.append(inter_Ts[i].detach().cpu().numpy())
                    file_list.append(f"{base_count:05}")
        os.makedirs(target_out_path, exist_ok=True)
        os.makedirs(target_normal_path, exist_ok=True)
        os.makedirs(target_inter_path, exist_ok=True)
        os.makedirs(out_weight_path, exist_ok=True)
        file_paths = [os.path.join(origin_inter_path, f) for f in os.listdir(origin_inter_path)]
        for file_path in file_paths:
            shutil.move(file_path, target_inter_path)
        df = pd.DataFrame(list(zip(file_list, inter_list)), columns=["img_name", "interpolate_position"])
        df.to_csv(os.path.join(out_weight_path, prompt_tails + ".csv"), index=False)
        with open(os.path.join(out_weight_path, prompt_tails + ".t"), 'wb') as f:
            torch.save(torch.cat((c1, c2), dim=0), f)
        shutil.rmtree(origin_out_path)


def generate(opt):
    devicestr = "cuda:" + str(opt.cuda)
    device = torch.device(devicestr) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model, sampler, clip, preprocess = init(device)

    if opt.classes == "imagenet":
        single_class = np.loadtxt('prompts/map_clsloc.txt', dtype=str)[:, 2].tolist()
        classes = list()
        exist_prompts = os.listdir("/opt/data/private/stable_diffusion/my_origin_data_inter")
        for _ in range(1000):
            i = random.choice(single_class)
            j = random.choice(single_class)
            if i == j or i.replace("_", " ") + "&" + j.replace("_", " ") in exist_prompts or get_single_hypersynsets(i) == "None" or get_single_hypersynsets(j) == "None":
                continue
            else:
                classes.append(i.replace("_", " ") + "&" + j.replace("_", " "))
    elif opt.classes == "cifar100":
        single_class = ('beaver', 'dolphin', 'otter', 'seal', 'whale',  # 水生哺乳动物
                        'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',  # 鱼
                        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',  # 花卉
                        'bottles', 'bowls', 'cans', 'cups', 'plates',  # 食品容器
                        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',  # 水果蔬菜
                        'clock', 'computer keyboard', 'lamp', 'telephone', 'television',  # 家用电器
                        'bed', 'chair', 'couch', 'table', 'wardrobe',  # 家用家具
                        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',  # 昆虫

                        'bridge', 'castle', 'house', 'road', 'skyscraper',  # 大型人造户外用品
                        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',  # 大型动物
                        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',  # 中型动物
                        'crab', 'lobster', 'snail', 'spider', 'worm',  # 无脊椎动物
                        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle')  # 爬行动物
        classes = list()
        for i in single_class:
            for j in single_class:

                if i == j:
                    continue
                else:
                    classes.append(i + "&" + j)
    elif opt.classes == "classes":
        single_class = ('camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers')
        classes = list()
        for i in single_class:
            for j in single_class:

                if i == j:
                    continue
                else:
                    classes.append(i + "&" + j)

    if (opt.fix):
        uc = model.get_learned_conditioning(4 * [""])
        size = (4, 512 // 8, 512 // 8)
        x_T_single = torch.randn(size, device=device)
        x_T = torch.stack([x_T_single for _ in range(4)])
    else:
        uc = None
        x_T = None

    thispart = opt.part
    total = opt.total
    part = len(classes) / total
    for i in trange(int(thispart * part), int((thispart + 1) * part), desc="part finish time"):
        if i >= len(classes):
            continue
        # single_class = classes[i] + " on the " + grounds[r] + "&" + classes[len(classes) - 1 - i] + " on the " + \
        #                grounds[s]

        if opt.option == "ablation":
            single_class = classes[i].replace("_", " ") + "&" + classes[len(classes) - 1 - i].replace("_", " ")
            getT_interpolate_ablation(s=50, out_path=opt.path, prompt_tails=single_class,
                                      model=model, sampler=sampler, clipmodel=clip,
                                      preprocess=preprocess, uc=uc, x_T=x_T, device=device)
        elif opt.option == "single":
            single_class = classes[i].replace("_", " ")
            getT_interpolate_single_check(s=50, out_path=opt.path, prompt_tail=single_class,
                                          model=model, sampler=sampler, clipmodel=clip,
                                          preprocess=preprocess, uc=uc, x_T=x_T, device=device)
        elif opt.option == "baseline":
            single_class = classes[i].replace("_", " ")
            getT_interpolate_baseline(s=50, out_path=opt.path, prompt_tails=single_class,
                                      model=model, sampler=sampler, clipmodel=clip,
                                      preprocess=preprocess, uc=uc, x_T=x_T, device=device)
        elif opt.option == "normal":
            single_class = classes[i].replace("_", " ") + "&" + classes[len(classes) - 1 - i].replace("_", " ")
            getT_interpolate(s=50, out_path=opt.path, prompt_tails=single_class,
                             model=model, sampler=sampler, clipmodel=clip,
                             preprocess=preprocess, uc=uc, x_T=x_T, device=device)
        elif opt.option == "multi":
            classes = list(classes)
            single_class = classes.pop(i).replace("_", " ")

            for _ in range(10):
                prompt_l = random.sample(classes, 3)
                prompt_l.insert(0, single_class)
                getT_interpolate_listPrompt(s=50, out_path=opt.path, prompt_tails=prompt_l,
                                            model=model, sampler=sampler, clipmodel=clip,
                                            preprocess=preprocess, uc=uc, x_T=x_T, device=device)
        elif opt.option == "classes":
            now_class = classes[i]
            # path = os.path.join("/opt/data/private/stable_diffusion", opt.path)
            getT_interpolate_classes(s=50, out_path=opt.path, prompt_tails=now_class,
                                     model=model, sampler=sampler, clipmodel=clip,
                                     preprocess=preprocess,uc=uc, x_T=x_T, device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default="0",
        help="GPU number"
    )
    parser.add_argument(
        "--fix",
        type=int,
        default=0,
        help="whether fix the uc and xT",
    )
    parser.add_argument(
        "--total",
        type=int,
        nargs="?",
        help="total divide",
        default="1"
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="part",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="genTs",
        help="path of image generation",
    )
    parser.add_argument(
        "--option",
        type=str,
        default="baseline"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="cifar100"
    )

    opt = parser.parse_args()
    return opt


def main():
    '''
        conda activate stable && cd /home/stablediffusion_v2
        '''
    # python getPerms_interpolate.py --path=my_class_test --part=2 --total=3 --cuda=3 --fix=0 --classes=imagenet --option=classes
    # python my_evaluation.py --part=1
    opt = parse_args()
    # model, sampler, clipmodel, preprocess = init(opt)

    generate(opt)


if __name__ == '__main__':
    main()
