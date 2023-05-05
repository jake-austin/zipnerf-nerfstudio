# zipnerf-nerfstudio

currently borked. hoping to be less borked, extenal collaborators welcome, feel free to make a PR and / or an issue :)

Notes:
proposal sampling seems to work when substituted in the nerfacto method for the mipnerf360 proposal loss that nerfacto was using. 
Multisampling / downweighting is where I suspect the issue is. 
Aiming still to have this run on 10GB vram, by using tinycudann's marginally smaller MLPs and decreasing the learning rate and batch size to be more reasonable.
Basing a lot of this off https://github.com/SuLvXiangXin/zipnerf-pytorch, just trying to get it further integrated with nerfstudio so we can officially list zipnerf as an external method for nerfstudio.
