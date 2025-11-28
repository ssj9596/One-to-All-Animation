import torch

class EDMPrecond():
    def __init__(self,      
        sigma_data      = 1,              # Expected standard deviation of the training data.
        sigma_min = 0,
        sigma_max = float('inf')
    ):
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def forward(self, model, x, sigma):
        dtype = x.dtype
        x = x.to(torch.float32) #(B, C, T, H, W)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = model((c_in * x).to(dtype), c_noise.flatten())[0]
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.edm_precond = EDMPrecond()

    def __call__(self, net, latents):
        #选择sigma，这里sigma的采样来源于p_{train}，是一个log norm的形式，即ln\sigma的分布是N(p_mean, p_std^2)
        rnd_normal = torch.randn([latents.shape[0]], device=latents.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2).reshape(-1, 1, 1, 1, 1)
        y = latents
        n = torch.randn_like(y) * (sigma.reshape(-1, 1, 1, 1, 1))
        # D_yn = net(y + n, sigma)[0]
        D_yn = self.edm_precond.forward(net, y+n, sigma)
        loss = weight * ((D_yn - y) ** 2)
        return loss