import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from data.util import res2img
import torch.nn.functional as F
from core.mylib import sobel_gradient
from core.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            loss_type='l1',
            conditional=True
    ):
        super(GaussianDiffusion, self).__init__()
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',  # 在sr3中，sqrt_alphas_cumprod是传入的noise_level
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod_1',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod_1',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    @torch.no_grad()
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def classifier_free_guidance_sample(self, w, x, noise_level, x_in):
        """
        once we have two model:condition and unconditional model,we have two noise to predict
        @param:w the weight to balance conditional and unconditonal
        (w+1)*model(x_t,t,y)-w*model(x_t,t,empty)
        """
        # condition_x = torch.cat([x_in['MS'], x_in['PAN']], dim=1)
        # zeros = torch.zeros_like(x_in['PAN'])
        # uncondition_x = torch.cat([x_in['MS'], zeros], dim=1)
        # noise_condition = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        # noise_uncondition = self.denoise_fn(torch.cat([uncondition_x, x], dim=1), noise_level)
        # return (w + 1) * noise_condition - w * noise_uncondition
        MS = x_in['MS']  # 输入信息变换为MS的高频输入，在网络中还会进一步转置卷积操作
        PAN = x_in['PAN']

        pan_concat = PAN.repeat(1, 4, 1, 1)  # Bsx8x64x64
        condition = torch.sub(pan_concat, MS)
        unconditional_conditional = torch.zeros_like(MS)

        noise_condition = self.denoise_fn(x, noise_level, condition)
        noise_uncondition = self.denoise_fn(x, noise_level, unconditional_conditional)
        return (w + 1) * noise_condition - w * noise_uncondition

    @torch.no_grad()
    def classifier_free_guidance_sample_pan2ms(self, w, x, noise_level, x_in):
        condition = x_in['PAN']
        unconditional_conditional = torch.zeros_like(condition)

        noise_condition = self.denoise_fn(x, noise_level, condition)
        noise_uncondition = self.denoise_fn(x, noise_level, unconditional_conditional)
        return (w + 1) * noise_condition - w * noise_uncondition

    @torch.no_grad()
    def classifier_free_guidance_sample_ms2pan(self, w, x, noise_level, x_in):
        condition = x_in['MS']
        unconditional_conditional = torch.zeros_like(condition)

        noise_condition = self.denoise_fn(x, noise_level, condition)
        noise_uncondition = self.denoise_fn(x, noise_level, unconditional_conditional)
        return (w + 1) * noise_condition - w * noise_uncondition

    @torch.no_grad()
    def p_mean_variance(self, x, t, clip_denoised=True, x_in=None, w=3.0):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        # 由于索引从T-1到0，sqrt_alphas_cumprod[999]等于sqrt_alphas_cumprod_prev[1000]，sqrt_alphas_cumprod_prev有1001个元素

        time_in = torch.tensor([t + 1] * batch_size, device=noise_level.device).view(batch_size, -1)
        assert time_in.shape == noise_level.shape

        if x_in is not None:
            # x_recon = self.predict_start_from_noise(  # 根据噪声得到x0
            #     x, t=t, noise=self.classifier_free_guidance_sample_ms2pan
            #     (w=w, x=x, noise_level=time_in, x_in=x_in))
            x_recon = self.denoise_fn(x, time_in, x_in['PAN'])

        if clip_denoised:
            x_recon = self.dynamic_clip(x_recon, is_static=True)

        model_mean, posterior_log_variance = self.q_posterior(  # 根据x0,xt,t真实分布公式直接计算即可
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def dynamic_clip(self, x_recon, is_static=True):
        if is_static:
            x_recon.clamp_(-1., 1.)
        else:
            s = torch.max(torch.abs(x_recon))
            s = s if s > 1 else 1.
            x_recon = x_recon / s
            print(s)
        return x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, w=3.0):
        # 在给定x,t和yt的条件下，获得yt-1的和均值和方差，值得注意的是估计的X0已经被归一化，通过公式计算yt-1
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, x_in=condition_x, w=w)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, w=3.0):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:  # 无条件x
            shape = x_in.shape
            img = torch.randn(shape, device=device)  # 随机高斯噪声
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i)  # timestamp从T到1,不断得到t-1时刻下的图像
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:  # super resolution的方法
            shape = x_in['MS'].shape
            img = torch.randn(shape, device=device)  # 高斯随机噪声，初始XT
            ret_img = img
            # ret_img = res2img(img, x_in['MS'])
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x_in, w=w)  # 在给定xt,t,condition，采样得到xt-1
                if i % sample_inter == 0:  # 用于展示的采样间隔，ret_img最终是一个列表，第一个元素为MS，最后一个结果则是T步去噪的结果
                    # ret_img = torch.cat([ret_img, res2img(img, x_in['MS'])], dim=0)
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img  # 一连串残差预测结果
        else:
            return ret_img[-1]

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, w=3.0):  # inference接口，会获取多个（8个）逐渐去噪的结果，并可以绘制一个过程图像
        # return self.p_sample_loop(x_in, continous, w=w)
        return self.sample_by_dpmsolver(x_in, w)

    @torch.no_grad()
    def sample_by_dpmsolver(self, x_in, w=3.0):
        # 值得注意的是，我们可以获得ret_img，记录每个step的去噪结果
        device = self.betas.device
        x_T = torch.randn(x_in['MS'].shape, device=device)
        condition = x_in['PAN']
        unconditional_condition = torch.zeros_like(condition)
        model = self.denoise_fn
        model_kwargs = {}
        guidance_scale = w

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="x_start",  # or "x_start" or "v" or "score"
            model_kwargs=model_kwargs,
            guidance_type="classifier-free",
            condition=condition,
            unconditional_condition=unconditional_condition,
            guidance_scale=guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                                correcting_x0_fn="dynamic_thresholding")

        x_sample = dpm_solver.sample(
            x_T,
            steps=20,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True
        )

        # ret_img = res2img(x_sample, x_in['MS'])

        return x_sample

    @torch.no_grad()
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):  # 扩散过程的加噪函数
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def epsilon_loss(self, noise, x_recon):
        loss = self.loss_func(noise, x_recon)
        return loss

    # def spatial_loss(self, x_hat, pan):
    #     # channel mean
    #     out2pan = torch.mean(x_hat, dim=1).unsqueeze(1)
    #     pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
    #     out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
    #     loss_pan_out = self.loss_func(pan_gradient_x, out2pan_gradient_x) + \
    #                    self.loss_func(pan_gradient_y, out2pan_gradient_y)
    #     return loss_pan_out
    #
    # def spectral_loss(self, x_hat, ms):
    #     [b, c, h, w] = ms.shape
    #     x_hat_down = F.interpolate(x_hat, size=h, mode='bicubic')
    #     loss = self.loss_func(x_hat_down, ms)
    #     return loss

    def classifier_free_guidance_train(self, x_in, p_uncond):
        """"
        classifier_free guidance ,jointly train a diffusion model with classifier-free guidance
        @param:p_uncond probability of unconditional training
        pan_guidance diffusion model
        """

        rand = torch.rand(1)
        MS = x_in['PAN_0']
        if rand > p_uncond:
            return MS
        else:
            return torch.zeros_like(MS)

    def p_losses_dynamic_x0(self, x_in, noise=None):
        # M2P,X0-XO(Xt,T,Condition),Unet输出的不是噪声而是X0,Unet直接利用MS条件预测PAN[-1,1]
        x_start = x_in['MS_0']
        [b, c, h, w] = x_start.shape
        # 随机选取一个timestamp索引1-T进行加噪
        time_in = torch.from_numpy(np.random.randint(1, self.num_timesteps + 1, size=b))
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            self.sqrt_alphas_cumprod_prev[time_in]).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        time_in = time_in.to(x_start.device).view(b, -1)

        x_noisy = self.q_sample(  # 对X0进行加噪得到xt，拓展[b,1]维度为[b,1,1,1],才可以和[b,c,h,w]的x_start,noise进行运算
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        cond = self.classifier_free_guidance_train(x_in, p_uncond=0.1)
        x_recon = self.denoise_fn(x_noisy, time_in, cond)

        loss = self.epsilon_loss(x_start, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses_dynamic_x0(x, *args, **kwargs)

    @torch.no_grad()
    def feats(self, x_in, time_in, noise=None):
        '''
            x: input image that you want to get features
            t: time step
            需要将MS部分加噪t步，然后送入网络得到PAN和加噪MS的编码与解码部分的信息（PAN2MS）
            每次获取一个时间步的就可以
        '''
        x_start = x_in['MS_0']
        [b, c, h, w] = x_start.shape
        time_in = torch.from_numpy(np.full(b, time_in))
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.array(self.sqrt_alphas_cumprod_prev[time_in])).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        time_in = time_in.to(x_start.device).view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))  # 需要预测的随机噪声
        x_noisy = self.q_sample(  # 对X0进行加噪得到xt，拓展[b,1]维度为[b,1,1,1],才可以和[b,c,h,w]的x_start,noise进行运算
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        cond = x_in['PAN_0']
        fe, fd = self.denoise_fn(x_noisy, time_in, cond, feat_need=True)

        return fe, fd


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep):
    if schedule == 'linear':
        scale = 1000 / n_timestep
        beta_start = scale * 1e-6  # 固定初始的beta和结束的beta,减少参数量
        beta_end = scale * 1e-2
        return np.linspace(
            beta_start, beta_end, n_timestep, dtype=np.float64
        )
    elif schedule == "cosine":
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(schedule)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
