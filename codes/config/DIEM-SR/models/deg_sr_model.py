import logging
from collections import OrderedDict
import random

import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = torch.clamp(input, 0, 1)
        output = (output * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


@MODEL_REGISTRY.register()
class DegSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["src", "tgt"]

        self.network_names = ["netDeg", "netSR", "netD1", "netD2","netDegInf"]
        self.networks = {}

        self.loss_names = [
            "lr_adv",
            "sr_adv",
            "sr_pix_trans",
            "sr_pix_sr",
            "sr_percep",
            "lr_quant",
            "lr_gauss",
            "noise_mean",
            "color"
        ]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())
        assert set(defined_network_names).issubset(set(self.network_names))
        
        for name in defined_network_names:
            setattr(self, name, self.build_network(nets_opt[name]))
            self.networks[name] = getattr(self, name)
            
        if self.is_train:
            train_opt = opt["train"]
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

            self.max_grad_norm = train_opt["max_grad_norm"]
            self.quant = Quantization()
            self.D_ratio = train_opt["D_ratio"]
            self.optim_sr = train_opt["optim_sr"]
            self.optim_deg = train_opt["optim_deg"]
            self.gray_dis = train_opt["gray_dis"]

            ## buffer
            self.fake_lr_buffer = ShuffleBuffer(train_opt["buffer_size"])
            self.fake_hr_buffer = ShuffleBuffer(train_opt["buffer_size"])

    def feed_data(self, data):

        self.syn_hr = data["tgt"].to(self.device)
        self.real_lr = data["src"].to(self.device)

    def deg_forward(self,step):
        (
            self.fake_real_lr,
            self.predicted_kernel,
            self.predicted_noise,
         ) = self.netDeg(self.syn_hr,step)

        if self.losses.get("sr_pix_trans"):
            self.fake_real_lr_quant = self.quant(self.fake_real_lr)
            self.syn_sr = self.netSR(self.fake_real_lr_quant)
    
    def sr_forward(self):
        if not self.optim_deg:
            (
                self.fake_real_lr,
                self.predicted_kernel,
                self.predicted_noise,
            ) = self.netDeg(self.syn_hr)

        self.fake_real_lr_quant = self.quant(self.fake_real_lr)
        deg_strength = self.netD1(self.fake_real_lr_quant)
        self.syn_sr = self.netSR(self.fake_real_lr_quant.detach(),deg_strength.detach())
    def optimize_trans_models(self, step, loss_dict):
        self.set_requires_grad(["netDeg"], True)
        self.deg_forward(step)
        loss_G = 0
        if self.losses.get("lr_adv"):
            self.set_requires_grad(["netD1"], False)
            if self.gray_dis:
                real = rgb_to_grayscale(self.real_lr)
                fake = rgb_to_grayscale(self.fake_real_lr)
            else:
                real = self.real_lr
                fake = self.fake_real_lr
            g1_adv_loss = self.calculate_gan_loss_G(
                self.netD1, self.losses["lr_adv"], real, fake)
            loss_dict["g1_adv"] = g1_adv_loss.item()
            loss_G += self.loss_weights["lr_adv"] * g1_adv_loss
        
        if self.losses.get("sr_pix_trans"):
            self.set_requires_grad(["netSR"], False)
            sr_pix = self.losses["sr_pix_trans"](self.syn_hr, self.syn_sr)
            loss_dict["sr_pix_trans"] = sr_pix.item()
            loss_G += self.loss_weights["sr_pix_trans"] * sr_pix
        
        if self.losses.get("noise_mean"):
            noise = self.predicted_noise
            noise_mean = (
                self.losses["noise_mean"](noise, torch.zeros_like(noise))
            )
            loss_dict["noise_mean"] = noise_mean.item()
            loss_G += self.loss_weights["noise_mean"] * noise_mean

        self.set_optimizer(names=["netDeg"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["netDeg"], self.max_grad_norm)
        self.set_optimizer(names=["netDeg"], operation="step")
        ## update D
        if step % self.D_ratio == 0:
            self.set_requires_grad(["netD1"], True)
            if self.gray_dis:
                real = rgb_to_grayscale(self.real_lr)
                fake = rgb_to_grayscale(self.fake_real_lr)
            else:
                real = self.real_lr
                fake = self.fake_real_lr
            loss_d1 = self.calculate_gan_loss_D(
                self.netD1, self.losses["lr_adv"],
                real, self.fake_lr_buffer.choose(fake)
            )
            loss_dict["d1_adv"] = loss_d1.item()
            loss_D = self.loss_weights["lr_adv"] * loss_d1
            self.optimizers["netD1"].zero_grad()
            loss_D.backward()
            self.clip_grad_norm(["netD1"], self.max_grad_norm)
            self.optimizers["netD1"].step()
        
        return loss_dict
    
    def optimize_sr_models(self, step, loss_dict):
        self.set_requires_grad(["netSR"], True)
        self.set_requires_grad(["netDeg"], False)
        self.sr_forward()
        loss_G = 0

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD2"], False)
            sr_adv_loss = self.calculate_gan_loss_G(
                self.netD2, self.losses["sr_adv"],
                self.syn_hr, self.syn_sr
            )
            loss_dict["sr_adv"] = sr_adv_loss.item()
            loss_G += self.loss_weights["sr_adv"] * sr_adv_loss
        
        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](
                self.syn_hr, self.syn_sr
            )
            loss_dict["sr_percep"] = sr_percep.item()
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style.item()
                loss_G += self.loss_weights["sr_percep"] * sr_style
            loss_G += self.loss_weights["sr_percep"] * sr_percep

        if self.losses.get("sr_pix_sr"):
            sr_pix = self.losses["sr_pix_sr"](self.syn_hr, self.syn_sr)
            loss_dict["sr_pix_sr"] = sr_pix.item()
            loss_G += self.loss_weights["sr_pix_sr"] * sr_pix

        self.set_optimizer(names=["netSR"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["netSR"], self.max_grad_norm)
        self.set_optimizer(names=["netSR"], operation="step")
        ## update D2
        if step % self.D_ratio == 0:
            if self.losses.get("sr_adv"):
                self.set_requires_grad(["netD2"], True)
                loss_d2 = self.calculate_gan_loss_D(
                    self.netD2, self.losses["sr_adv"],
                    self.syn_hr, self.fake_hr_buffer.choose(self.syn_sr)
                )
                loss_dict["d2_adv"] = loss_d2.item()
                loss_D = self.loss_weights["sr_adv"] * loss_d2
                self.optimizers["netD2"].zero_grad()
                loss_D.backward()
                self.clip_grad_norm(["netD2"], self.max_grad_norm)
                self.optimizers["netD2"].step()
        
        return loss_dict


    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        # optimize trans
        if self.optim_deg:
            loss_dict = self.optimize_trans_models(step, loss_dict)

        # optimize SR
        if self.optim_sr:
            loss_dict = self.optimize_sr_models(step, loss_dict)

        self.log_dict = loss_dict
    
    def calculate_gan_loss_D(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake.detach())
        d_pred_real = netD(real)
        mean_tensor = torch.mean(d_pred_fake, dim=1, keepdim=True)
        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)

        return (loss_real + loss_fake) / 2

    def calculate_gan_loss_G(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake)
        loss_real = criterion(d_pred_fake, True, is_disc=False)

        return loss_real

    def test(self, test_data, crop_size=None):
        self.src = test_data["src"].to(self.device)
        if test_data.get("tgt") is not None:
            self.tgt = test_data["tgt"].to(self.device)
            
        self.set_network_state(["netSR"], "eval")
        with torch.no_grad():
            if crop_size is None:
                deg_strength = self.netD1(self.src)
                self.fake_tgt = self.netSR(self.src,deg_strength.detach())
            else:
                self.fake_tgt = self.crop_test(self.src, crop_size)
        self.set_network_state(["netSR"], "train")

        if hasattr(self, "netDeg"):
            self.set_network_state(["netDeg"], "eval")
            if hasattr(self, "tgt"):
                with torch.no_grad():
                    self.fake_lr = self.netDeg(self.tgt)[0]
            self.set_network_state(["netDeg"], "train")

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.src.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_tgt.detach()[0].float().cpu()
        if hasattr(self, "fake_lr"):
            out_dict["fake_lr"] = self.fake_lr.detach()[0].float().cpu()
        return out_dict
    
    def crop_test(self, lr, crop_size):
        b, c, h, w = lr.shape
        scale = self.opt["scale"]

        h_start = list(range(0, h-crop_size, crop_size))
        w_start = list(range(0, w-crop_size, crop_size))

        sr1 = torch.zeros(b, c, int(h*scale), int(w* scale), device=self.device) - 1
        for hs in h_start:
            for ws in w_start:
                lr_patch = lr[:, :, hs: hs+crop_size, ws: ws+crop_size]
                deg_strength = self.netD1(lr_patch)
                sr_patch = self.netSR(lr_patch,deg_strength.detach())

                sr1[:, :, 
                    int(hs*scale):int((hs+crop_size)*scale),
                    int(ws*scale):int((ws+crop_size)*scale)
                ] = sr_patch
        
        h_end = list(range(h, crop_size, -crop_size))
        w_end = list(range(w, crop_size, -crop_size))

        sr2 = torch.zeros(b, c, int(h*scale), int(w* scale), device=self.device) - 1
        for hd in h_end:
            for wd in w_end:
                lr_patch = lr[:, :, hd-crop_size:hd, wd-crop_size:wd]
                deg_strength = self.netD1(lr_patch)
                sr_patch = self.netSR(lr_patch,deg_strength.detach())
                sr2[:, :, 
                    int((hd-crop_size)*scale):int(hd*scale),
                    int((wd-crop_size)*scale):int(wd*scale)
                ] = sr_patch

        mask1 = (
            (sr1 == -1).float() * 0 + 
            (sr2 == -1).float() * 1 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        mask2 = (
            (sr1 == -1).float() * 1 + 
            (sr2 == -1).float() * 0 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        sr = mask1 * sr1 + mask2 * sr2

        return sr


class ShuffleBuffer():
    """Random choose previous generated images or ones produced by the latest generators.
    :param buffer_size: the size of image buffer
    :type buffer_size: int
    """

    def __init__(self, buffer_size):
        """Initialize the ImagePool class.
        :param buffer_size: the size of image buffer
        :type buffer_size: int
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []

    def choose(self, images, prob=0.5):
        """Return an image from the pool.
        :param images: the latest generated images from the generator
        :type images: list
        :param prob: probability (0~1) of return previous images from buffer
        :type prob: float
        :return: Return images from the buffer
        :rtype: list
        """
        if self.buffer_size == 0:
            return  images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_imgs += 1
            else:
                p = random.uniform(0, 1)
                if p < prob:
                    idx = random.randint(0, self.buffer_size - 1)
                    stored_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(stored_image)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images